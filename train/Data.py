import glob
import os
import random

from audiomentations import Compose, RoomSimulator
import datasets
import librosa
import numpy as np
import pedalboard
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent, detect_silence
from scipy import signal
import torch

from transformers import AutoFeatureExtractor

from get_base_audio import get_base_audio

class Dataset():
    def __init__(self, lang, valid_count=None, streaming=False):
        self.encoded_dataset = datasets.load_dataset(f'../train/data/data_{lang}.py', streaming=streaming)

        if valid_count:
            self.encoded_dataset["validation"] = self.encoded_dataset["validation"].select(range(min(valid_count, len(self.encoded_dataset["validation"]))))

class CustomDataCollator:
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"
    def __init__(self, input_sec, audio_model, debug):
        assert 2 < input_sec, "input_sec is too short"
        self.input_sec = input_sec
        self.audio_model = audio_model

        self.laughter_audio_dirs = [r"D:\datasets\vocals\vs_release_16k\audio_16k", \
                           r"D:\datasets\laughterscape_ver1.0\ver1.0\denoised", \
                            r"D:\datasets\manually_collected\laughter",]
        self.laughter_audio_probs = [0.5, 0.45, 0.025, 0.025] # last one is for synthetic crowd laughters
        
        self.base_audio_dir = r"D:\spotify_podcast_dataset"
        self.base_audio_dir2 = r"D:\datasets\various_audioset_wav"
        self.base_audio_probs = [0.8, 0.2]

        # self.laughter_like_noise_dir = r"D:\datasets\vocals\vs_release_16k\audio_16k\others"
        self.music_fragment_dir = r"D:\datasets\IRMAS\IRMAS-TrainingData\IRMAS-TrainingData"
        self.IRMAS_music_dirs = [r"D:\datasets\IRMAS\IRMAS-TestingData-Part1\IRMAS-TestingData-Part1\Part1",
                                 r"D:\datasets\IRMAS\IRMAS-TestingData-Part2\IRMAS-TestingData-Part2\IRTestingData-Part2",
                                 r"D:\datasets\IRMAS\IRMAS-TestingData-Part3\IRMAS-TestingData-Part3\Part3"]
        self.misc_dir = r"D:\datasets\ESC-50-master\ESC-50-master\audio"

        self.sr = 16000
        self.sr_mul = self.sr // 1000
        train_1 = glob.glob(self.laughter_audio_dirs[0] + "/*.wav")
        train_2 = glob.glob(self.laughter_audio_dirs[1] + "/*.wav")
        train_3 = glob.glob(self.laughter_audio_dirs[2] + "/*.mp3")+glob.glob(self.laughter_audio_dirs[2] + "/*.wav") +\
                    glob.glob(self.laughter_audio_dirs[2] + "/*/*.mp3")+glob.glob(self.laughter_audio_dirs[2] + "/*/*.wav") #crowd laughters
        TRAIN_RATIO = 0.98
        self.train_audios = [train_1[:int(len(train_1)*TRAIN_RATIO)], train_2[:int(len(train_2)*TRAIN_RATIO)], train_3]
        self.eval_audios = [train_1[int(len(train_1)*TRAIN_RATIO):], train_2[int(len(train_2)*TRAIN_RATIO):]]
        self.resample_to = [None if int(librosa.load(random.choice(self.train_audios[0]), sr=None, mono=True, duration=1)[1]) == self.sr else self.sr, 
                            None if int(librosa.load(random.choice(self.train_audios[1]), sr=None, mono=True, duration=1)[1]) == self.sr else self.sr,
                            self.sr]
        # assert self.resample_to[0] == None, f"sample rate of dataset 1 should be 16000." # もともと16kの場合余計なリサンプリングを避けたい。librosa.load(sr=None)の場合2倍早くなった
        # assert self.resample_to[1] == self.sr, f"sample rate of dataset 2 should be 24000."
        # assert len(self.train_audios[0]) and len(self.train_audios[1]), "train audio not found in some dir"
        assert len(self.eval_audios[0]) and len(self.eval_audios[1]), "eval audio not found in some dir"

        self.base_audios = glob.glob(self.base_audio_dir + "/*")
        self.base_audios2 = glob.glob(self.base_audio_dir2 + "/*")
        assert self.base_audios, "no base_audios found"
        assert self.base_audios2, "no base_audios2 found"
        self.train_base_audios = self.base_audios[:int(len(self.base_audios)*TRAIN_RATIO)]
        self.eval_base_audios = self.base_audios[int(len(self.base_audios)*TRAIN_RATIO):]
        self.train_base_audios2 = self.base_audios2[:int(len(self.base_audios2)*TRAIN_RATIO)]
        self.eval_base_audios2 = self.base_audios2[int(len(self.base_audios2)*TRAIN_RATIO):]

        # self.laughter_like_noise_audios = glob.glob(self.laughter_like_noise_dir + "/*.wav")
        # assert self.laughter_like_noise_audios, "no laughter_like_noise found"
        # assert int(librosa.load(random.choice(self.laughter_like_noise_audios), sr=None, mono=True, duration=1)[1]) == self.sr

        # self.music_fragment_audios = glob.glob(self.music_fragment_dir + "/*/*.wav")
        self.music_fragment_audios = glob.glob(self.music_fragment_dir + "/voi/*.wav")
        assert len(self.music_fragment_audios) > 500, "seems too few music audios"

        self.music_audios = []
        # get max 3 music fragments from each music
        for music_dir in self.IRMAS_music_dirs:
            audio_paths = glob.glob(glob.escape(music_dir) + "/*.wav")
            assert audio_paths, f"no music found in {music_dir}"
            audio_paths = [os.path.splitext(audio_path)[0].rsplit("-", 1)[0] for audio_path in audio_paths]
            audio_paths = list(set(audio_paths))
            for audio_path in audio_paths:
                music_paths = glob.glob(glob.escape(audio_path) + "*.wav")
                assert music_paths, f"no music found in {audio_path}"            
                self.music_audios += random.sample(music_paths, min(3, len(music_paths)))
        assert len(self.music_audios) > 300, "seems too few music audios" 

        self.misc_audios = glob.glob(glob.escape(self.misc_dir)+ "/*.wav")
        assert len(self.misc_audios) > 1500, "no enough misc found"

        self.MAX_TRY = 100
        self.MAX_LAUGH_COUNT = 4
        self.MODEL_OUT_FRAME_LEN = 349 # about 49 frames per second

        self.min_cutoff_freq = 2000

        self.curriculum_learning = True if not debug else False
        self.current_iteration = 0
        self.curriculum_learning_100_percent_iteration = 2000

        self.debug = debug

        if audio_model:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model)
            print("audio max length", int(self.feature_extractor.sampling_rate * self.input_sec))

    def is_inappropriate_audio(self, array):
        if array.shape[0] < self.sr * 1: # remove very short audio
            # print("too short", os.path.basename(audio_path))
            return True
        if max(array) < .1: # remove slient audio. should check min too but looks the same as -max
            # print("too silent", os.path.basename(audio_path))
            return True
        return False

    def add_laughter_like_noise(self, base_audio, base_len):
        # add Sigh, Cough, ThroatClearing, Sneeze, Sniff
        if random.random() < 1/6: # choose from [No noise, Sigh, Cough, ThroatClearing, Sneeze, Sniff]
            return base_audio
        noise_audio_path = random.choice(self.laughter_like_noise_audios)
        noise_array = librosa.load(noise_audio_path, sr=None, mono=True)[0]
        noise_len = len(noise_array)
        start_idx = random.randint(-noise_len//2, base_len-noise_len//2)
        noise_array *= random.uniform(.1, 1)
        base_audio[max(0, start_idx): min(base_len, noise_len+start_idx)] += noise_array[max(-start_idx, 0): min(noise_len, base_len-start_idx)]
        return base_audio
    
    def add_sound_to_no_laugh_part(self, base_audio, base_len, label, vol):
        # find label==0 continues for some time and add short fragment sound
        no_laugh_part_duration = self.sr * 5
        start_frame_idx = -1
        for idx, frame in enumerate(label):
            if frame == 0 and start_frame_idx == -1:
                start_frame_idx = idx
            elif start_frame_idx != -1 and (frame == 1 or idx == base_len-1):
                if idx-start_frame_idx > no_laugh_part_duration:
                    if random.random() < .75:
                        self.debug and print(f"add_sound_to_no_laugh_part {start_frame_idx}~{idx}")
                        margin = random.randint(no_laugh_part_duration, idx-start_frame_idx)
                        start_idx = random.randint(start_frame_idx, idx-margin)
                        base_audio = self.add_short_audio_fragment_to_specific_pos(self.misc_audios,
                                                                    base_audio, 
                                                                    base_len, 
                                                                    start_idx,
                                                                    margin,
                                                                    vol)
                    # if random.random() < .75:
                    #     self.debug and print(f"add_sound_to_no_laugh_part {start_frame_idx}~{idx}")
                    #     margin = random.randint(no_laugh_part_duration, idx-start_frame_idx)
                    #     start_idx = random.randint(start_frame_idx, idx-margin)
                    #     base_audio = self.add_short_audio_fragment_to_specific_pos(self.laughter_like_noise_audios,
                    #                                                 base_audio, 
                    #                                                 base_len, 
                    #                                                 start_idx,
                    #                                                 margin,
                    #                                                 vol)
                start_frame_idx = -1
        return base_audio

    # add short (<3s) music fragment to base_audio
    def add_short_audio_fragment_to_specific_pos(self, audios, base_audio, base_len, start_idx, margin, vol):
        music_audio_path = random.choice(audios)
        music_array = librosa.load(music_audio_path, sr=self.sr, mono=True)[0]

        music_array = self.change_pitch_and_speed(music_array)
        
        _end_idx = min(len(music_array), margin)
        music_array = music_array[:_end_idx]
        music_len = len(music_array)
        music_array *= vol
        # add fade in/out if start/end is in base_audio
        fade_len = min(music_len//2, int(self.sr*random.uniform(0.01,0.5))) # 0~0.5秒
        # if 0 < start_idx:
        music_array[:fade_len] *= np.linspace(0, 1, fade_len)
        # if start_idx+music_len < base_len:
        music_array[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        # fade base_audio where music exists
        fade_intensity=random.uniform(.01, .95)
        base_audio[start_idx: start_idx+music_len] *= fade_intensity

        base_audio[start_idx: start_idx+music_len] += music_array
        return base_audio
    
    # add short (<3s) music fragment to base_audio
    def add_short_music_fragment(self, base_audio, base_len):
        for _ in range(1):
            music_audio_path = random.choice(self.music_fragment_audios)
            music_array = librosa.load(music_audio_path, sr=self.sr, mono=True)[0]
            music_len = len(music_array)
            start_idx = random.randint(-music_len//2, base_len-music_len//2)
            music_array *= random.uniform(.2, 1)
            # add fade in/out if start/end is in base_audio
            fade_len = min(music_len//2, int(self.sr*random.uniform(0.01,0.5))) # 0~0.5秒
            if 0 < start_idx:
                music_array[:fade_len] *= np.linspace(0, 1, fade_len)
            if start_idx+music_len < base_len:
                music_array[-fade_len:] *= np.linspace(1, 0, fade_len)
            
            # fade base_audio where music exists
            fade_intensity=random.uniform(.01, .95)
            base_audio[max(0, start_idx): min(base_len, music_len+start_idx)] *= fade_intensity

            base_audio[max(0, start_idx): min(base_len, music_len+start_idx)] += music_array[max(-start_idx, 0): min(music_len, base_len-start_idx)]
        return base_audio
    
    def add_short_misc_sound(self, base_audio, base_len):
        audio_path = random.choice(self.misc_audios)
        audio_array = librosa.load(audio_path, sr=self.sr, mono=True)[0]
        audio_len = len(audio_array)
        start_idx = random.randint(-audio_len//2, base_len-audio_len//2)
        audio_array *= random.uniform(.2, 1)

        base_audio[max(0, start_idx): min(base_len, audio_len+start_idx)] += audio_array[max(-start_idx, 0): min(audio_len, base_len-start_idx)]
        return base_audio
    
    def add_music(self, base_audio, base_len):
        music_audio_path = random.choice(self.music_audios)
        music_array = librosa.load(music_audio_path, sr=self.sr, mono=True, duration=20)[0]
        music_len = len(music_array)
        # decide random start point if music is longer than input_sec
        input_len = self.sr*self.input_sec
        music_array *= random.uniform(.7, 1)
        if music_len > input_len:
            start_idx = random.randint(0, music_len-input_len)          
            music_array = music_array[start_idx: start_idx+input_len]
            base_audio += music_array
        else:
            start_idx = random.randint(0, base_len-music_len)
            base_audio[start_idx: start_idx+music_len] += music_array
        return base_audio
    
    def low_pass_filter(self, array, cutoff_freq=1000):
        nyquist_freq = 0.5 * self.sr
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = signal.butter(5, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, array)
    
    def compressor(self, array, ratio):
        # array -= array.mean()
        board = pedalboard.Pedalboard([
            pedalboard.Compressor(threshold_db=-30, 
                                ratio=ratio, 
                                attack_ms = 1.0,
                                release_ms = 100)
        ])
        array = board(array, self.sr)
        return array
        # return librosa.util.normalize(array)
    
    def custom_amplituder_small_portion(self, array, sr=16000, mul_fac=5):
        # 32767 is max value of signed short
        dub_audio = AudioSegment(
                    (array*32767).astype("int16").tobytes(), 
                    sample_width=2, 
                    frame_rate=16000, 
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

    def reverb(self, array, \
               room_size=None, damping=None, wet_level=None, dry_level=None, width=None, freeze_mode=None, \
               min_absorption_value=None, max_size_x=None, max_size_y=None, max_size_z=None, \
                lib_prob=None):
        prob = random.random() < 0.5 if lib_prob is None else lib_prob
        if prob < 0.5:
            if room_size is None: room_size = random.random()
            if damping is None: damping = random.uniform(.1, 1.)
            if wet_level is None: wet_level = random.uniform(.0, .5)
            dry_level = 1; width = 1.0; freeze_mode = 0.0
            board = pedalboard.Pedalboard([
                    pedalboard.Reverb(room_size = room_size,
                                    damping = damping,
                                    wet_level = wet_level,
                                    dry_level = dry_level,
                                    width = width,
                                    freeze_mode = freeze_mode),
                ])
            return board(array, self.sr)
        else:
            if min_absorption_value is None: min_absorption_value = random.random()
            if max_size_x is None: max_size_x = random.uniform(5, 100)
            if max_size_y is None: max_size_y = random.uniform(5, 100)
            if max_size_z is None: max_size_z = random.uniform(5, 100)
            min_mic_distance = 1; max_mic_distance = 50
            reverb = Compose([RoomSimulator(leave_length_unchanged=True, 
                                                    use_ray_tracing=False,
                                                    p=1,min_target_rt60=1,
                                                    min_absorption_value=min_absorption_value,
                                                    min_mic_distance=min_mic_distance,
                                                    max_mic_distance=max_mic_distance,
                                                    max_order=2,
                                                    max_size_x=max_size_x,
                                                    max_size_y=max_size_y,
                                                    max_size_z=max_size_z)])
            try:
                return reverb(array, sample_rate=self.sr)
            except ValueError as e:
                print(e)
                return array

    def crop_nonsilent_section(self, array):
        if self.is_inappropriate_audio(array):
            return None
        
        # array = librosa.util.normalize(array)
        dub_audio = AudioSegment(
            (array*32767).astype("int16").tobytes(), 
            sample_width=2, 
            frame_rate=16000, 
            channels=1,
            )
        dub_audio = normalize(dub_audio)

        nonsilent_section = detect_nonsilent(dub_audio, min_silence_len=270, silence_thresh=-35)

        if not nonsilent_section:
            return None
        # 基本各音声の最初の笑いを取る。後ろのほうは息を吸う音などが入っていることが多い
        for nons_sec in nonsilent_section:
            start_idx, end_idx = int(nons_sec[0]*self.sr_mul), int(nons_sec[1]*self.sr_mul)

            # 短い音声は笑いでないことが多い(0.3秒)
            if end_idx-start_idx < self.sr*0.3:
                continue
            # 0.1秒くらい前に挿入し、笑いが寸断されるのを防止
            short_margin = int(self.sr * .1)
            start_idx = max(0, start_idx-short_margin)

            laugh_audio = array[start_idx: end_idx]
            return laugh_audio
        return None

    def load_laugh(self, chosen_dataset, split):
        trial_count = 0
        while trial_count < self.MAX_TRY:
            trial_count += 1
            audio_path = random.choice(self.train_audios[chosen_dataset] if split == "train" else self.eval_audios[1 if chosen_dataset>=2 else chosen_dataset])
            # max duraion is 10 seconds but usually shorter
            laugh_audio = librosa.load(audio_path, sr=self.resample_to[chosen_dataset], mono=True, duration=10)[0]
            if random.random() < .05:
                try:
                    laugh_audio = self.reverb(laugh_audio)
                    self.debug and print(f" reverb to laugh")
                except: pass
            if random.random() < .15:
                try:
                    ratio = random.uniform(1, 10)
                    laugh_audio = self.compressor(laugh_audio, ratio)
                    self.debug and print(f" compressor to laugh {ratio}")
                except: pass
            if random.random() < .05:
                try:
                    cutoff_freq = random.uniform(self.min_cutoff_freq,3000)
                    laugh_audio = self.low_pass_filter(laugh_audio, cutoff_freq=cutoff_freq)
                    self.debug and print(f" low_pass_filter to laugh {cutoff_freq}Hz")
                except: pass
            laugh_audio = self.crop_nonsilent_section(laugh_audio)
            if laugh_audio is not None:
                self.debug and print(f"load_laugh {os.path.basename(audio_path)}")
                return laugh_audio
        raise ValueError(f"laugh audio not found in {trial_count} trials")
    
    def change_pitch_and_speed(self, array):
        # to float32
        array = array.astype(np.float32)
        if random.random() < .5:
            pitch = random.uniform(-10,10)
            pitch_c = pedalboard.Pedalboard(
                [pedalboard.PitchShift(semitones=pitch)]
            )
            array = pitch_c(array, sample_rate=self.sr)
        if random.random() < .5:
            speed = min(max(0.75, random.gauss(1,.3)), 2)
            array = pedalboard.time_stretch(array, samplerate=self.sr, stretch_factor=speed)[0]
        return array
    
    def make_crowd_laugh(self, split):
        trial_count = 0
        default_start_idx = self.sr # 1 second
        while trial_count < self.MAX_TRY:
            trial_count += 1
            chosen_dataset = random.randint(0, 1)
            crowd_laugh_audio = np.zeros(self.sr*self.input_sec)
            num_of_laughing_people = random.randint(5, 30)

            room_size = random.random()
            damping = random.uniform(.1, 1.)
            wet_level = random.uniform(.0, .3)
            min_absorption_value = random.random()
            max_size_x = random.uniform(5, 100)
            max_size_y = random.uniform(5, 100)
            max_size_z = random.uniform(5, 100)
            reverb_lib = random.random() < .5

            for _ in range(num_of_laughing_people):
                laugh_audio = self.load_laugh(chosen_dataset, split)
                laugh_audio = self.reverb(laugh_audio, room_size=room_size, damping=damping, wet_level=wet_level, \
                                          min_absorption_value=min_absorption_value, max_size_x=max_size_x, max_size_y=max_size_y, max_size_z=max_size_z, lib_prob=reverb_lib)
                comp_ratio = random.uniform(1, 7)
                laugh_audio = self.compressor(laugh_audio, comp_ratio)
                low_pass_cutoff = random.uniform(self.min_cutoff_freq,8000)
                laugh_audio = self.low_pass_filter(laugh_audio, cutoff_freq=low_pass_cutoff)
                start_idx = max(0, int(random.gauss(default_start_idx,default_start_idx//5)))
                end_idx = min(start_idx+len(laugh_audio), len(crowd_laugh_audio))
                crowd_laugh_audio[start_idx: end_idx] += laugh_audio[:min(len(crowd_laugh_audio)-start_idx, len(laugh_audio))]
            crowd_laugh_audio = librosa.util.normalize(crowd_laugh_audio)
            crowd_laugh_audio = self.crop_nonsilent_section(crowd_laugh_audio)
            if crowd_laugh_audio is None:
                continue
            # fade out if too long
            if len(crowd_laugh_audio) > self.sr*5:
                crowd_laugh_audio = crowd_laugh_audio[:int(self.sr*random.uniform(3., 6.))]
            fade_len = int(self.sr*random.uniform(.5, 1.5))
            if fade_len < len(crowd_laugh_audio):
                crowd_laugh_audio[len(crowd_laugh_audio)-fade_len:] *= np.linspace(1, 0, fade_len)
            return crowd_laugh_audio
        raise ValueError(f"laugh audio not found in {trial_count} trials")
    
    def get_laugh_sample(self, base_audio, label, laugh_audio, base_len, weak_fade):
        laugh_len = len(laugh_audio)

        margin = laugh_len // 3
        insert_idx = random.randint(-margin, base_len-min(self.sr*1, laugh_len))
        # 挿入位置がbase_audioの外側の場合、重なる部分が無音でないことを確認。無音の場合はmarginを取らずに再度ランダムに挿入位置を決定
        if insert_idx < 0:
            if max(laugh_audio[-insert_idx:]) < .1:
                insert_idx = random.randint(0, base_len-min(self.sr*1, laugh_len))
        elif insert_idx > base_len-min(self.sr*1, laugh_len):
            if max(laugh_audio[:base_len-insert_idx]) < .1:
                insert_idx = random.randint(0, base_len-min(self.sr*1, laugh_len))

        # fade in/out laugh_audio
        fade_len = int(self.sr*random.uniform(0.01,0.1))
        if random.random() > .5 and laugh_len > (fade_len*2):
        # if laugh_len > (fade_len*2):
            laugh_audio[:fade_len] *= np.linspace(0, 1, fade_len)
            laugh_audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        fade_intensity=random.uniform(.01, .95)
        fade_start_offset = random.randint(0, int(min(self.sr, laugh_len*0.1)))
        if base_len < insert_idx+fade_start_offset: fade_start_offset = 0
        if insert_idx > fade_len:
            base_audio[insert_idx-fade_len+fade_start_offset: insert_idx+fade_start_offset] *= np.linspace(1, fade_intensity, fade_len)
        base_audio[max(0, insert_idx+fade_start_offset): min(insert_idx+laugh_len, base_len)] *= fade_intensity
        if insert_idx+laugh_len < base_len-fade_len:
            base_audio[insert_idx+laugh_len: insert_idx+laugh_len+fade_len] *= np.linspace(fade_intensity, 1, fade_len)
        

        # change volume of laugh_audio randomly
        laugh_audio *= random.uniform(.3, 1.2)
        # prevent laughter from becoming inaudible
        laugh_avg = np.mean(abs(laugh_audio[max(-insert_idx, 0): min(laugh_len, base_len-insert_idx)]))
        base_avg = np.mean(abs(base_audio[max(0, insert_idx): min(base_len, laugh_len+insert_idx)]))
        if laugh_avg < base_avg:
            laugh_audio *= (base_avg/laugh_avg)#*1.1

        base_audio[max(0, insert_idx): min(base_len, laugh_len+insert_idx)] += laugh_audio[max(-insert_idx, 0): min(laugh_len, base_len-insert_idx)]
        label[max(0, insert_idx): min(base_len, laugh_len+insert_idx)] = 1

        return base_audio, label
        # curr_laugh_count += 1
    
    def getitem(self, split, laugh_count=None):
        chosen_base_idx = random.choices([0, 1], weights=self.base_audio_probs)[0]
        base_audio = get_base_audio(([self.train_base_audios, self.train_base_audios2][chosen_base_idx]
                                            if split=="train" else [self.eval_base_audios, self.eval_base_audios2][chosen_base_idx]),
                                        self.sr,
                                        self.input_sec,
                                        [self.base_audio_dir, self.base_audio_dir2][chosen_base_idx], # to detect base_audio dataset
                                        (min(1, (self.current_iteration)*.85/self.curriculum_learning_100_percent_iteration) if (self.curriculum_learning and split == "train") else 1),
                                        self.debug,
                                        )
        
        base_len = len(base_audio)
        # add cough or throat clear sound not to be confused with laughter
        # base_audio = self.add_laughter_like_noise(base_audio, base_len)

        label = np.zeros(base_len, dtype=int)
        laugh_count = random.randint(0, self.MAX_LAUGH_COUNT) if laugh_count is None else laugh_count

        # if laugh_count == 0 and random.random() < .15:
        if random.random() < .1:
            if random.random() < .2:
                base_audio = self.add_short_music_fragment(base_audio, base_len)
            else:
                base_audio = self.add_music(base_audio, base_len)

        if self.curriculum_learning and split == "train":
            base_audio *= min(1, self.current_iteration/self.curriculum_learning_100_percent_iteration)            
        
        curr_laugh_count = 0
        while curr_laugh_count < laugh_count:
            is_synthetic_laugh = False
            chosen_dataset = random.choices([0, 1, 2, 3], weights=self.laughter_audio_probs)[0]
            if chosen_dataset == 3:
                is_synthetic_laugh = True
            
            if is_synthetic_laugh:
                laugh_audio = self.make_crowd_laugh(split)
                self.debug and print("make_crowd_laugh")
            else:
                laugh_audio = self.load_laugh(chosen_dataset, split)
                if random.random() < .2:
                    laugh_audio = self.change_pitch_and_speed(laugh_audio)
            assert laugh_audio is not None, "laugh_audio is None"
            
            base_audio, label = self.get_laugh_sample(base_audio,
                                                        label,
                                                        laugh_audio,
                                                        base_len,
                                                        is_synthetic_laugh,
                                                        )
            curr_laugh_count += 1

        base_audio = self.add_sound_to_no_laugh_part(base_audio, base_len, label,
                                                        min(1, self.current_iteration/self.curriculum_learning_100_percent_iteration) if self.curriculum_learning else 1)

        # centering
        base_audio -= base_audio.mean()

        # data augmentation
        if split == "train":
            data_arg_prop = min(1, self.current_iteration/self.curriculum_learning_100_percent_iteration) if self.curriculum_learning else 1
        else:
            data_arg_prop = 1

        if random.random() < .02 * data_arg_prop:
            self.debug and print("downsample quantization bit rate")
            base_audio = np.round(base_audio, 2)
        if random.random() < .12 * data_arg_prop:
            self.debug and print("add reverb")
            base_audio = self.reverb(base_audio)
        if random.random() < .15 * data_arg_prop:
            ratio = random.uniform(1, 10)
            self.debug and print(f"compressor with ratio {ratio}")
            base_audio = self.compressor(base_audio, ratio=ratio)
        if random.random() < .08 * data_arg_prop:
            cutoff_freq = random.randint(self.min_cutoff_freq, 3000)
            self.debug and print(f"low pass filter with cutoff_freq {cutoff_freq}")
            base_audio = self.low_pass_filter(base_audio, cutoff_freq=cutoff_freq)
            base_audio *= 1.2
        
        # randomly shorten audio and padding with 0 to the end
        if random.random() < .005:
            random_pos = random.randint(self.sr*2, base_len)
            self.debug and print(f"shorten audio to {random_pos} and padding with 0 to the end")
            base_audio = base_audio[:random_pos]
            base_audio = np.append(base_audio, np.zeros(base_len-len(base_audio)))
            label = label[:random_pos]
            label = np.append(label, np.zeros(base_len-len(label)))
            assert len(base_audio) == len(label) == base_len, f"{random_pos=}, {len(base_audio)=}, {len(label)=}, {base_len=}"
                
        if not (split == "train" and \
                self.curriculum_learning and \
                self.current_iteration < int(self.curriculum_learning_100_percent_iteration*0.75)):
            base_audio = self.custom_amplituder_small_portion(base_audio)

        # sd.play(base_audio, sr, blocking=True)
        # audio_arrays.append(base_audio)
        audio_array = base_audio

        label = label[np.linspace(0, base_len-1, self.MODEL_OUT_FRAME_LEN, dtype=int)]
        # labels["labels"].append(label)
        # FOR GILLICL MODEL. get center idx value
        # print(label[np.linspace(0, label.shape[0]-1, 40, dtype=int)][15:25])

        # visualize waveform and label with rectangle
        # import sounddevice as sd
        # import matplotlib.pyplot as plt
        # plt.plot(base_audio)
        # width = base_len / self.MODEL_OUT_FRAME_LEN
        # for i, l in enumerate(label):
        #     if l:
        #         plt.axvspan(i*width, (i+1)*width, ymax=0.5, color='green', alpha=.7)
        # sd.play(base_audio, self.sr, loop=True)
        # plt.show()

        if self.debug:
            import sounddevice as sd
            print(label)
            sd.play(audio_array, self.sr, blocking=True)

        return audio_array, label

    def __call__(self, features):
        audio_arrays = []
        labels = {"labels": []}

        for feat in features:
            audio_array, label = self.getitem(feat["split"])
            audio_arrays.append(audio_array)
            labels["labels"].append(label)
        
        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sr,
            max_length=int(self.feature_extractor.sampling_rate * self.input_sec),
            truncation=True,
            padding="max_length",#True,
            return_attention_mask=False,
            return_tensors=self.return_tensors,
        )

        labels["labels"] = torch.tensor(np.array(labels["labels"])).type(torch.int)
        
        # audio_features = torch.tensor(audio_arrays).type(torch.float16)
        # labels["labels"] = torch.tensor(np.array(labels["labels"])).type(torch.int)

        self.current_iteration += 1
        
        return dict(**audio_features, **labels)
