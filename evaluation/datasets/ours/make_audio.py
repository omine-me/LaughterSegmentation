import glob
import os
import json
import librosa
import soundfile as sf
import audioread

major_idx = "0"
sr = 16000

base_dir = os.path.join(r"D:\datasets\en\podcast\podcast\podcasts-audio", major_idx)

output_dir = os.path.join(r".", "audio", "non_laugh")
output_dir = os.path.join(r".", "gt", "non_laugh")
# output_dir = os.path.join(r".", "audio", "laugh")

for non_laugh_json in glob.glob("./gt/_existence_only/non_laugh/*.json"):
# for non_laugh_json in glob.glob("./gt/_existence_only/laugh/*.json"):
    # out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(non_laugh_json))[0]+".wav")
    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(non_laugh_json))[0]+".json")
    with open(non_laugh_json, "w", encoding="utf-8") as f:
        json.dump({}, f)


    # if os.path.exists(out_path):
    #     continue

    # with open(non_laugh_json, "r") as f:
    #     json_data = json.load(f)
    # assert len(json_data) == 1
    # start_sec = json_data["0"]["start_sec_in_source"]
    # if start_sec < 0:
    #     raise Exception(f"start_sec is negative in {non_laugh_json}")
    # duration = json_data["0"]["duration"]
    # end_sec = start_sec + duration

    # show_name = "show_" + os.path.basename(non_laugh_json).split("_")[1]
    # minor_idx = show_name[6]
    # episode_name = os.path.basename(non_laugh_json).split("_")[2]
    # audio_path = os.path.join(base_dir, minor_idx, show_name, episode_name+".ogg")

    # audio = librosa.load(audio_path, sr=sr, offset=start_sec, duration=duration, mono=True)[0]

    # with audioread.audio_open(audio_path) as f:
    #     audio_length_s = float(int(f.duration))
    # if audio_length_s < end_sec:
    #     raise Exception(f"audio duration is shorter than end_sec written in {non_laugh_json}")

    # sf.write(out_path, audio, sr)
    # # assert False