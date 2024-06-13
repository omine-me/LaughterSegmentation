# Evaluation
We evaluate our paper with four evaluation datasets. `datasets` directory contains evaluation datasets. `models` directory contains the inference results and inference codes for each model.

If you want to get the same results as in the paper, get audio datasets first as written below, and run `evaluate.py` in the same directory. Comment out the end of the file accordingly. (We will eventually improve the script so that audio files are not needed.)

If you wish to evaluate your own model or infer the results from scratch, follow these steps.

1. Get datasets
    - Some data is not in this repository due to rights issues, however, you can download them.
    - Petridis2013
        - This is an evaluation dataset from [MAHNOB Laughter Database](https://mahnob-db.eu/laughter). You need to request account to download audio.
        1. Download the 130 audio files for which Spontaneous Laughter Filter is Yes and save it in `evaluation/datasets/Petridis2013/audio`.
        - The annotation data is created as follows. Since the Ground Truth data is already contained in the `gt` directory, you basically do not need to do these things.
        1. Download anotation data from [here](https://mahnob-db.eu/laughter/media/uploads/voicedunvoicedlaughter_annotations.xls) and [here](https://mahnob-db.eu/laughter/media/uploads/annotations.xls).
        1. Copy VoicedLaughter, Speechlaughter, and PosedLaughter into one csv without any header.
        1. Save it as `annotations.csv` in `evaluation/datasets/Petridis2013/original_anotation_data`.
        1. Run `extract_laughter.py` in the dataset directory.
    - McCowan2005
        - This is an evaluation dataset from [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/). You need to download audio.
        1. Download Headset mix audios in list of SC(ES2004, ES2014, IS1009, TS3003, TS3007, EN2002) of "Full-corpus partition of meeting" in this [page](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml). Download from [here](https://groups.inf.ed.ac.uk/ami/download/) and save all to ./audio without any directory hierarchy.
        - The annotation data is collected as follows. Since the Ground Truth data is already contained in the `gt` directory, you basically do not need to do these things.
        1. Download "AMI manual annotations v1.6.2" from [https://groups.inf.ed.ac.uk/ami/download/]()
        1. Extract and copy `words` directory to `./original_anotation_data`.
        1. Run `extract_laughter.py` in the dataset directory.
    - Gillick2021
        - This is an evaluation dataset based on [Gillick2021](https://github.com/jrgillick/laughter-detection/tree/master) & [AudioSet](https://groups.inf.ed.ac.uk/ami/corpus/). You need to download audio.
        1. Download `eval_segments.csv`, `balanced_train_segments.csv`, and `unbalanced_train_segments.csv` from [AudioSet website](https://research.google.com/audioset/download.html).
        1. Download `clean_distractor_annotations.csv`, and `clean_laughter_annotations.csv` from [GitHub](https://github.com/jrgillick/laughter-detection/tree/master/data/audioset/annotations) (Alternatively, it is automatically included when the repository is cloned, as described below.).
        1. Download audio from YouTube. VideoID is written in csv from GitHub, and time is written in csv from AudioSet. Or you can refer to `gt` directory to see which audio you need (time info isn't written). For various reasons, some video is not available.
        - The annotation data is created as follows. Since the Ground Truth data is already contained in the `gt` directory, you basically do not need to do these things.
        1. Get laughter segmentation data from `clean_laughter_annotations.csv`, and convert it in json format. For data from `clean_distractor_annotations.csv`, simply generate an empty json file. See the `gt` directory for details.
    - Ours
        - This is our evaluation dataset based on Spotify Podcast Dataset. You need to download audio.
        1. Download audio from [Here](https://drive.google.com/drive/folders/1dNBOscaXeBakDuvpLPGGExyfZgHqjw0F?usp=sharing), extract and save it to `audio` directory. Make sure the `audio` directory contains `laugh` and `non_laugh` directories.
        - The annotation data is created manually. Contains 201 each of data with and without laughter. See paper for details.
1. Infer with models. Run `infer.py` in `models/{model_name}` directory. Comment out the end of the file accordingly. See paper for details on each model. You need [jrgillick/laughter-detection](https://github.com/jrgillick/laughter-detection) to infer previous study models. Run `git clone https://github.com/jrgillick/laughter-detection.git` in the top directory (where requirements.txt exists).
1. Run `evaluate.py` in this directory. Comment out the end of the file accordingly.
