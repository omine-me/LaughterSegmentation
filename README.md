# Laughter Segmentation

## Overview
You can extract a exact segment of laughter from various talking audio using trained model and code. You can also train your own model.

Code, annotations, and model are described in the following paper:
[Taisei Omine, Kenta Akita, and Reiji Tsuruno, "Robust Laughter Segmentation with Automatic Diverse Data Synthesis", Interspeech 2024.]() (To be published in a few months)

## Installation
```sh
git clone https://github.com/omine-me/LaughterSegmentation.git
cd LaughterSegmentation
python -m pip install -r requirements.txt
# ↓ Depends on your environment. See https://pytorch.org/get-started/locally/
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
Run in Venv environment is recommended. Also, download `model.safetensors` from [Huggingface](https://huggingface.co/omine-me/LaughterSegmentation/tree/main) (1.26 GB) and place it in `models` directory and make sure the name is `model.safetensors`.

Tested on Windows 11 with GeForce RTX 2060 SUPER.

## Usage
1. Prepare audio file.
1. Open Terminal and go to the directory where `inference.py` is located.
1. Run `python inference.py --audio_path audio.wav`. You have to change *audio.wav* to your own audio path. You can use common audio format like `mp3`, `wav`, `opus`, etc. 16kHz wav audio is faster.
1. If you want to change output directory, use  `--output_dir` option. If you want to use your own model, use `--model_path` option.
1. Result will be saved in output directory in json format. To visualize the results, you can use [this site](https://omine-me.github.io/AudioDatasetChecker/compare.html) (not perfect because it's for debugging).

## Training
Added about 1 or 2 weeks later.

## Evaluation (Includes our evaluation dataset)
Read [README](/evaluation/README.md) in evaluavtion directory.

## License
This repository is MIT-licensed, but [the publicly available trained model](https://huggingface.co/omine-me/LaughterSegmentation/tree/main) is currently available for research use only.

## Contact
Use Issues or reach out my [X(Twitter)](https://x.com/mineBeReal).
