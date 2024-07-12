# get nonlaugh audio from yamnet score

import os
import glob
import json
import random

import librosa
import sounddevice as sd
import h5py

import audioread

LAUGHTER_IDX = 13
BABY_LAUGHER_IDX = 14
GIGGLE_IDX = 15
SNICKER_IDX = 16
BELLY_LAUGH_IDX = 17
CHUCKLE_IDX = 18

SPEECH_IDX = 0
CHILD_SPEECH_IDX = 1
CONVERSATION_IDX = 2
NARRATION_IDX = 3

# laugh_threshold = .01 #1%
laugh_threshold = .005
max_laugh_threshold = .05 #5%
speech_threshold = .01 #1%

MAX_TRY_FILES = 100
MAX_TRY_IN_FILE = 10
YAMNET_DIR = r"D:\datasets\en\podcast\podcast\yamnet\score"

def get_base_audio(audios, sr, duration, base_audio_dir="", curriculum_learning_rate=1.0, debug=False):
    if "audioset" in base_audio_dir:
        trial = 0
        while trial < MAX_TRY_FILES:
            audio_path = random.choice(audios)
            with audioread.audio_open(audio_path) as f:
                audio_length = float(f.duration)
            if audio_length < duration:
                trial += 1
                continue

            audio_array = librosa.load(audio_path, sr=sr, duration=duration, mono=True)[0]
            if len(audio_array) < (duration*.97)*sr:
                print("BROKEN",audio_path)
                trial += 1
                continue
            debug and print(f"non-laugh audio: {os.path.basename(audio_path)}")
            return audio_array
        raise ValueError("Take too many trials to get non-laugh audio.")
    else:
        trial = 0
        while trial < MAX_TRY_FILES:
            audio_path = random.choice(audios)
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            # major_idx = basename[5]
            # minor_idx = basename[6]
            # show_name = "show_" + basename.split("_")[1]
            # episode_name = basename.split("_")[2]
            # yamnet_path = os.path.join(YAMNET_DIR, major_idx, minor_idx, show_name, episode_name+".h5")
            yamnet_path = os.path.join(YAMNET_DIR, basename+".h5")
            
            with h5py.File(yamnet_path, "r") as f:
                interval = float(f["score"]["axis1"][1])
                scores = f["score"]["block0_values"]
                # speech_probs = scores[:, SPEECH_IDX] + scores[:, CHILD_SPEECH_IDX] + scores[:, CONVERSATION_IDX] + scores[:, NARRATION_IDX]
                laughter_probs = scores[:, LAUGHTER_IDX] + scores[:, BABY_LAUGHER_IDX]+ scores[:, GIGGLE_IDX] + scores[:, SNICKER_IDX] + scores[:, BELLY_LAUGH_IDX] + scores[:, CHUCKLE_IDX]
                assert len(laughter_probs) > int(duration/interval), f"{audio_path=} is too short"
                
                inner_trial = 0
                while inner_trial < MAX_TRY_IN_FILE:
                    start_idx = random.randint(0, len(laughter_probs)-int((duration+1)/interval))
                    # if curriculum_learning_rate < 1.0:
                    #     curr_laugh_threshold = max(laugh_threshold, max_laugh_threshold * curriculum_learning_rate)
                    # elif curriculum_learning_rate == 1.0:
                    #     curr_laugh_threshold = laugh_threshold
                    # else:
                    #     raise ValueError(f"{curriculum_learning_rate=} should be in [0, 1]")
                    curr_laugh_threshold = laugh_threshold
                    if (laughter_probs[start_idx:start_idx+int(duration/interval)] < curr_laugh_threshold).all():# and\
                        # (speech_probs[start_idx:start_idx+int(duration/interval)] > speech_threshold).sum() > int(duration/interval)*.5:
                        # sd.play(librosa.load(audio_path, sr=sr, mono=True, offset=start_idx*interval, duration=duration)[0], sr, blocking=True)
                        try:
                            ar = librosa.load(audio_path, sr=sr, mono=True, offset=start_idx*interval, duration=duration)[0]
                        except Exception as e:
                            print(e)
                            print(audio_path, sr, start_idx*interval, duration)
                        debug and print(f"non-laugh audio: {os.path.basename(audio_path)}, start_time: {start_idx*interval}")
                        return ar
                    else:
                        inner_trial += 1
            trial += 1
        raise ValueError("Take too many trials to get non-laugh audio.")
