import os
import subprocess
import sys

from tgt_to_json import main as tgt_to_json

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

def main(referenced_paper):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', "datasets", referenced_paper))
    from evaluation_list import evaluation_list

    AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', '..', "datasets", referenced_paper, "audio")
    VENV_DIR = os.environ['VIRTUAL_ENV']
    laughter_detection_repo = os.path.join(os.path.dirname(__file__), '..', '..', '..', "laughter-detection")
    
    out_dir = os.path.join(os.path.dirname(__file__), referenced_paper, "textgrid")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for audio_id in evaluation_list:
        audio_file_path = get_audio_file_path(referenced_paper, AUDIO_DIR, audio_id)
        
        # Currently Windows only
        subprocess.run([os.path.join(VENV_DIR, "Scripts", "activate.bat"), "&",
                        sys.executable, os.path.join(laughter_detection_repo, "segment_laughter.py"),
                        "--config", "resnet_with_augmentation_omine",
                        "--model_path", 'checkpoints/omine',
                        "--input_audio_file", audio_file_path,
                        "--output_dir", out_dir,
                        "--save_to_textgrid", "True",
                        "--save_to_audio_files", "False",
                        ], cwd=laughter_detection_repo)
    
    tgt_to_json(referenced_paper)

if __name__ == "__main__":
    main("Gillick2021")
    # main("ours")
    # main("McCowan2005")
    # main("Petridis2013")
