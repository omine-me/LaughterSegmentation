import os, datetime, argparse, glob, json, librosa, sys
import os.path as osp
# import sounddevice as sd
import numpy as np
import pedalboard

from pydub import AudioSegment
from pydub.silence import detect_silence

import sys
sys.path.append(os.path.join('..', '..'))
from _utils.utils import concat_close, remove_short

def get_audio_file_path(referenced_paper, audio_dir, audio_id):
    if referenced_paper == "McCowan2005":
        return os.path.join(audio_dir, audio_id+".Mix-Headset.wav")
    elif referenced_paper == "Liu2022":
        raise NotImplementedError("Liu2022 is not implemented yet.")
    elif referenced_paper == "ours":
        # show_name, episode_name = audio_id.split("/")
        # return os.path.join(audio_dir, show_name+"_"+episode_name+".wav")
        if audio_id.split("_")[-2] == "non":
            suffix = "non_laugh"
        else:
            suffix = "laugh"
        return os.path.join(audio_dir, suffix, audio_id+".wav")
    elif referenced_paper == "Petridis2013":
        return os.path.join(audio_dir, audio_id+".wav")
    elif referenced_paper == "Gillick2021":
        if os.path.exists(os.path.join(audio_dir, audio_id+".wav")):
            return os.path.join(audio_dir, audio_id+".wav")
        elif os.path.exists(os.path.join(audio_dir, audio_id+".opus")):
            return os.path.join(audio_dir, audio_id+".opus")
        elif os.path.exists(os.path.join(audio_dir, audio_id+".mp3")):
            return os.path.join(audio_dir, audio_id+".mp3")
        raise ValueError("Audio file not found: {}".format(audio_id))
    else:
        raise ValueError("Unknown referenced_paper: {}".format(referenced_paper))

def merge_events(event_lists):
    merged_events = {}
    merged_event_idx = 0
    has_merged = False
    for event_list in event_lists:
        for event in event_list.values():
            if not merged_events:
                # If merged_events is empty, add the first event
                merged_events[str(merged_event_idx)] = event.copy()
                merged_event_idx += 1
            else:
                merged = False
                for merged_event in merged_events.values():
                    if event["start_sec"] <= merged_event["end_sec"] and event["end_sec"] >= merged_event["start_sec"]:
                        # Events overlap, merge them
                        merged_event["start_sec"] = min(event["start_sec"], merged_event["start_sec"])
                        merged_event["end_sec"] = max(event["end_sec"], merged_event["end_sec"])
                        merged = True
                        has_merged = True
                        # break
                if not merged:
                    # If the event does not overlap with any merged event, add it to merged_events
                    merged_events[str(merged_event_idx)] = event.copy()
                    merged_event_idx += 1
    if has_merged:
        merged_events = merge_events([merged_events])
    merged_events = sorted(merged_events.values(), key=lambda x: x["start_sec"])
    merged_events = {str(idx): val for idx, val in enumerate(merged_events)}
    return merged_events

from scipy import signal
#バターワースフィルタ（バンドパス）
def bandpass(x, samplerate, fp=np.array([1000,3000]), fs=np.array([1000,3000]), gpass=3, gstop=40):
    fn = samplerate / 2 #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band") #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x) #信号に対してフィルタをかける
    return y
def low_pass_filter(array, sr, cutoff_freq=2500):
    nyquist_freq = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(5, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, array)

def compressor(array, sr):
    array -= array.mean()

    # Create a Pedalboard
    board = pedalboard.Pedalboard([
        pedalboard.Compressor(threshold_db=-30, ratio=10, )
    ])

    # Process the audio through the pedalboard
    effected = board(array, sr)
    return librosa.util.normalize(effected)

def custom_amplituder_small_portion(array, sr, mul_fac=5):
    # 32767 is max value of signed short
    dub_audio = AudioSegment(
                (array*32767).astype("int16").tobytes(), 
                sample_width=2, 
                frame_rate=sr, 
                channels=1,
                )
    
    dub_audio = dub_audio.set_frame_rate(sr)
    silent_section = detect_silence(dub_audio, min_silence_len=270, silence_thresh=-35)

    sr_mul = sr // 1000
    for sec in silent_section:
        fade_len = int(sr*.15) # 0.15秒
        if (sec[1]-sec[0])*sr_mul > (fade_len*2):
            array[sec[0]*sr_mul: sec[0]*sr_mul + fade_len] *= np.linspace(1, mul_fac, fade_len)
            array[sec[0]*sr_mul + fade_len: sec[1]*sr_mul - fade_len] *= mul_fac
            if sec[1]*sr_mul < len(array):
                array[sec[1]*sr_mul - fade_len: sec[1]*sr_mul] *= np.linspace(mul_fac, 1, fade_len)
        else:
            array[sec[0]*sr_mul: sec[1]*sr_mul] *= mul_fac
    array = librosa.util.normalize(array)
    return array

def main(referenced_paper, input_sec, batch_size=1, debug=False):
    date = str(datetime.datetime.now()).split(".")[0].replace(" ","_").replace(":","-")

    from transformers.trainer_utils import set_seed
    import torch
    import safetensors
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', "train")
    sys.path.append(MODEL_DIR)
    from model import Model
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', "datasets", referenced_paper))
    from evaluation_list import evaluation_list

    lang = "en"
    audio_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

    sr = 16000
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    over_lap_sec = 2.
    assert input_sec > over_lap_sec

    audiodir = os.path.join(os.path.dirname(__file__), '..', '..', "datasets", referenced_paper, "audio")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    model = Model(audio_model_name, device, sr).to(device)
    """
    # model.load_state_dict(torch.load(f"{MODEL_DIR}/out/{train_id}/checkpoint-2000/pytorch_model.bin"))
    path = glob.glob(f"{MODEL_DIR}/out/*{glob.escape(train_id)}/pytorch_model.bin")
    if len(path) == 0:
        # path = glob.glob(f"{MODEL_DIR}/out/*{glob.escape(train_id)}/model.safetensors")
        path = glob.glob(f"{MODEL_DIR}/out/*{glob.escape(train_id)}/model.safetensors")
        assert len(path) == 1, f"{MODEL_DIR}/out/*{glob.escape(train_id)}/model.safetensors {path}"
        state_dict = safetensors.torch.load_file(path[0], device.index if device.type=="cuda" else "cpu")
    elif len(path) == 1:
        state_dict = torch.load(path[0])
    else:
        raise ValueError(f"More than one model found: {path}")
    """
    state_dict = safetensors.torch.load_file(os.path.join("..", "..", "..", "models", "model.safetensors"), device.index if device.type=="cuda" else "cpu")
    # state_dict = torch.load(model_path) # use when model is .bin format
    model.load_state_dict(state_dict)

    # inference
    # laughter_dir = osp.join(".", referenced_paper, "yamnet_predicted")

    out_dir = osp.join(".", referenced_paper)
    if os.path.exists(out_dir):
        if glob.glob(osp.join(out_dir, "*.json")):
            raise ValueError(f"Output directory {out_dir} is not empty.")
    else:
        os.makedirs(out_dir)
    
    model.eval()
    with torch.no_grad():
        for done_count, audio_id in enumerate(evaluation_list):
            audio_file_path = get_audio_file_path(referenced_paper, audiodir, audio_id)
            # audio_file_path = audio_id
            basename = osp.splitext(osp.basename(audio_file_path))[0]
            out_file = osp.join(out_dir, basename+".json")

            laughter = {}
            laughter_idx = 0

            audio_array = librosa.load(audio_file_path, sr=sr, mono=True)[0]
            # audio_array = low_pass_filter(audio_array, sr)

            # audio_array = compressor(audio_array, sr=sr)
            audio_array = custom_amplituder_small_portion(audio_array, sr)

            # get each array of 7 sec 
            for array_idx in range(0, len(audio_array), int(sr*(input_sec-over_lap_sec))*batch_size):
                batched_arrays = []
                should_break = False
                for batch_idx in range(batch_size):
                    array = audio_array[array_idx+batch_idx*int(sr*(input_sec-over_lap_sec)): array_idx+batch_idx*int(sr*(input_sec-over_lap_sec))+sr*input_sec]
                    # array = custom_amplituder_small_portion(array, sr)
                    if len(array) < sr*input_sec:
                        # add 0 to the end of array
                        array = np.append(array, np.zeros(sr*input_sec-len(array)))
                        should_break = True
                    batched_arrays.append(array)
                    if should_break:
                        break

                input_values = torch.from_numpy(np.array(batched_arrays)).type(torch.FloatTensor)
                # input_values = torch.unsqueeze(input_values, dim=0)
                outputs = model(input_values=input_values)

                logits = outputs[1]#.squeeze(dim=0)
            
                #  --- predict ends ---

                preds = torch.sigmoid(logits.to(torch.float32))

                for batch_idx, pred in enumerate(preds): # each batch
                    frame_pred = list(map(round, pred.cpu().tolist(), [3]*len(pred)))

                    # 0, 1 に変える
                    frame_pred = (np.array(frame_pred)>=0.5).astype(int)

                    batch_start_sec = (array_idx+batch_idx*int(sr*(input_sec-over_lap_sec)))/float(sr)
                    frame_count = len(frame_pred)
                    start_idx = None
                    end_idx = None
                    status = "not_laughing"
                    for idx, frame in enumerate(frame_pred):
                        if frame == 1:
                            if status == "not_laughing":
                                start_idx = idx
                                status = "laughing"
                            
                            # 最後のフレームまで笑いの場合
                            if status == "laughing" and idx == frame_count-1:
                                laughter[str(laughter_idx)] = {
                                    "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                    "end_sec": batch_start_sec + input_sec,
                                    # "is_batch_end": True,
                                }
                                laughter_idx += 1
                                start_idx = None
                                end_idx = None
                        elif frame == 0:
                            # end of laughter
                            if status == "laughing":
                                end_idx = idx
                                status = "not_laughing"
                                if start_idx == 0:
                                    laughter[str(laughter_idx)] = {
                                    # "is_batch_start": True,
                                    "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                    "end_sec": batch_start_sec + (input_sec/frame_count)*end_idx,
                                    }
                                else:
                                    laughter[str(laughter_idx)] = {
                                        "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                        "end_sec": batch_start_sec + (input_sec/frame_count)*end_idx,
                                    }
                                laughter_idx += 1
                                start_idx = None
                                end_idx = None
            
            if over_lap_sec > 0.:
                laughter = merge_events([laughter])

            with open(out_file, mode='w', encoding="utf-8") as f:
                laughter = concat_close(laughter, 0.2)
                laughter = remove_short(laughter, 0.2)
                json.dump(laughter, f)
            if done_count % (len(evaluation_list)//4) == 0:
                print(f"Done: {done_count+1}/{len(evaluation_list)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default="")

    args = parser.parse_args()

    if args.dataset:
        assert args.dataset in ["Gillick2021", "Petridis2013", "McCowan2005", "Liu2022", "ours"]
        print(args.dataset)
        main(args.dataset, 7, 10, args.debug)
    else:
        main("Gillick2021", 7, 10, args.debug)
        # main("Petridis2013", 7, 10, args.debug)
        # main("McCowan2005", 7, 10, args.debug)
        # main("Liu2022", 7, 10, args.debug)
        # main("ours", 7, 10, args.debug)
