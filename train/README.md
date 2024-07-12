# Training

## Reproduction of the paper
Tested on Windows 10 with GeForce RTX 2080 Ti.

1. Get datasets
    - Datasets for laughter samples
        - [VocalSound](https://github.com/YuanGongND/vocalsound) (16kHz Version)
        - [Laughterscape](https://sites.google.com/site/shinnosuketakamichi/research-topics/laughter_corpus)
        - Manually collected samples
            - Refer to `manually_collected_samples.md` in `dataset_list` directory.
    - Datasets for Base audio
        - [Spotify Podcast Dataset](https://podcastsdataset.byspotify.com/)
            - This dataset is no longer accessible. However, the episode list is in `./dataset_list/BaseAudio_Spotify_list.csv` and you can access to the podcast audio from URLs. We apologize for the inconvenience. We also converted them to wav 16kz audio for faster training.
        - [AudioSet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
            - The used YouTube videoID list is in `./dataset_list/BaseAudio_AudioSet_list.csv`. Refer to `balanced_train_segments.csv` from [AudioSet website](https://research.google.com/audioset/download.html) for more information such as start and end time of audio segments. We selected videos that did not contain tags related to laughter (`"Laughter(/m/01j3sz)", "Baby laughter(/t/dd00001)", "Giggle(/m/07r660_)", "Snicker(/m/07s04w4)", "Belly laugh(/m/07sq110)", "Chuckle, chortle(/m/07rgt08)"`). We also converted them to wav 16kz audio for faster training.
    - Datasets for noise
        <!-- - [VocalSound](https://github.com/YuanGongND/vocalsound) (16kHz Version) (same as above) -->
        - [IRMAS](https://www.upf.edu/web/mtg/irmas)
            - Download the whole IRMAS dataset from download section.
        - [ESC-50](https://github.com/karolpiczak/ESC-50)
            - Download the dataset from GitHub and exclude laughter samples (40 files with filenames ending in `-26`).
    
1. Save them in any folder and rewrite the path in `Data.py` (lines 37 to 52). Wav 16kz audio is preferable for the faster training.

1. Extract YAMNet scores from Spotify Podcast Dataset audio.
    1. This process was originally unnecessary, but now that we do not have access to the dataset, it is necessary to prepare a YAMNET score.
    1. Learn how to get your YAMNet score [here](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_visualization.ipynb).
    1. Save them in a single directory as an h5 file with the same name as the audio file. ([Example h5 file](https://drive.google.com/file/d/192_Jjv2nPOSKV4zub4M8WXA-cm2y3-SC/view?usp=sharing))
        1. A h5 file should contain the `"score"` group and have at least the `"axis1"` and `"block0_values"` datasets in it.
        1. `["score"]["axis1"]` is like and timestamp and is an 1D array every 0.48 seconds. e.g. [0.0, 0.48. 0.96, ...]
        1. `["score"]["block0_values"]` is a 2D array of the same length as axis1 x 521. It is a so-called score, which infers the likelihood of 521 different audio events every 0.48 seconds.
    1. Rewrite `YAMNET_DIR` in `get_base_audio.py` (line 33).

1. In some cases, an older version of the `soundfile` library is used. This causes errors when loading mp3 files, so consider running these commands in advance. More information can be found [here](https://stackoverflow.com/questions/75813603/python-working-with-sound-librosa-and-pyrubberband-conflict).
    ```Batchfile
    python -m pip uninstall pysoundfile
    python -m pip uninstall soundfile
    python -m pip install soundfile
    ```

1. Download FFmpeg and add it to the PATH as well to prevent audio loading errors.

1. Run `python train.py --id {training_id}` in `train` directory and the model will be in the `out` directory. Change `{training_id}` for any name. You can also use `--checkpoint` option to resume a training.

## Train your own model
You can train your own model using your own data, not only laughter. Basically, you can do this by rewriting `Data.py`. We may add more later.
