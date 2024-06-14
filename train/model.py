import librosa
from scipy.signal import find_peaks
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForAudioFrameClassification


class DiceLoss(nn.Module):
    ### refered: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    ### https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/comments

    # https://arxiv.org/abs/1606.04797
    # https://github.com/faustomilletari/VNet/blob/584c1502f369f2547e7ed384f9cc25beb09f65e1/pyLayer.py
    # dice loss提唱者の公式実装の模様
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.thresh = 0.5

    def forward(self, inputs, targets, smooth=0.00001):
        inputs = torch.sigmoid(inputs)
        batch_size = targets.shape[0]
        intersection = (inputs * targets).sum(dim=1)
        dice = ((2. * intersection) /
                (inputs.sum(dim=1) + targets.sum(dim=1) + smooth))
        dice = dice.mean() ### 本家はsum()
        
        return (1 - dice)


class Model(torch.nn.Module):
    def __init__(self, audio_model_name: str, device: str, sr: int):
        super().__init__()
        # These are needed for model parallel
        # self.is_parallelizable = True
        # self.model_parallel = True
        self.audio_name = audio_model_name
        self.device = device
        self.sr = sr

        if "wav2vec" in audio_model_name:
            self.audio_model = Wav2Vec2ForAudioFrameClassification.from_pretrained(audio_model_name, num_labels=1, problem_type="single_label_classification").to(device)
        else:
            raise Exception(f"{audio_model_name} is not supported")

        self.subprocesses_read_bytes_by_pids = {}
    
    def kill_subprocess_randomly(self):
        # return if whole pc memory usage is less than 80%
        import psutil        

        current_process = psutil.Process()
        children = current_process.children(recursive=True)

        if (len(children) < 27) and psutil.virtual_memory().percent < 70:
            return

        import random
        if random.random() > 0.3:
            if random.random() < 0.3:
                self.subprocesses_read_bytes_by_pids = {child.pid: child.io_counters()[2] for child in current_process.children(recursive=True)}
            return

        # pick one that are not using cpu, and not accessing disk
        children = [child for child in children if child.cpu_percent() == 0]
        children = [child for child in children if child.io_counters()[2] == self.subprocesses_read_bytes_by_pids.get(child.pid, 0)]
        
        # kill one
        if len(children) > 0:
            child = children[-1]
            print(f"Killing subprocess {child.pid}")
            # doubble check
            if child.cpu_percent() == 0:
                child.kill()
        
        self.subprocesses_read_bytes_by_pids = {child.pid: child.io_counters()[2] for child in current_process.children(recursive=True)}


    def forward(
        self,
        input_values: torch.Tensor = None, # audio
        mask_time_indices: torch.FloatTensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        labels=None,
        **kwargs,
    ):
        try:
            if labels is None:
                self.kill_subprocess_randomly()
        except:
            pass

        audio_outputs = self.audio_model(
            input_values=input_values.to(self.device),
        )
            
        logits = audio_outputs["logits"].squeeze(dim=2)
        if labels is not None:
            # DiceLoss    
            loss_func = DiceLoss().to(self.device)
            bce_loss_func = torch.nn.BCEWithLogitsLoss().to(self.device)
            labels = labels.to(torch.float)

            loss = loss_func(logits, labels.to(self.device)) + bce_loss_func(logits, labels.to(self.device))

            outputs = (loss, logits)
        else:
            outputs = (None, logits)

        return outputs

    def beat_loss(self, array, gt, eval):
        fp_frames = ((eval-gt)>0.5).to(torch.float)
        # array to 0 if fp_frames is False.
        fp_frames = fp_frames.repeat_interleave(round(array.shape[1]/349), dim=1)[:, :array.shape[1]]
        array *= fp_frames

        hop_length = 256
        onset_envelope = librosa.onset.onset_strength(y=array.to('cpu').detach().numpy().copy(), sr=self.sr, hop_length=hop_length)
        onset_envelope = librosa.util.normalize(onset_envelope)

        fp_len = torch.sum(fp_frames).item()
        if fp_len == 0:
            return .0
        loss = 0
        for i in range(array.shape[0]):
            peaks, _ = find_peaks(onset_envelope[i], height=.15)
            loss += len(peaks)/fp_len*50

        return loss
