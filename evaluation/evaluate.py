import json
import glob
import os
import random

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import audioread
from statistics import mean, stdev
from sklearn.metrics import mean_squared_error

from confidence_intervals import evaluate_with_conf_int
from decimal import Decimal, ROUND_HALF_UP

from _utils.utils import concat_close, remove_short

input_sec = 10#7
frame_wise_metrics = {}

datasets_by_dynamic_generation = ["McCowan2005", "Petridis2013"]

def should_skip_sample(laughter_segment):
    if "not_a_laugh" in laughter_segment and laughter_segment["not_a_laugh"]:
        return True
    if laughter_segment["end_sec"] - laughter_segment["start_sec"] <= 0.0:
        return True
    return False

def get_audio_path(audio_dir, basename):
    audio_file = os.path.join(audio_dir, basename+".wav")
    if not os.path.exists(audio_file):
        audio_file = os.path.join(audio_dir, basename+".opus")
    if not os.path.exists(audio_file):
        audio_file = os.path.join(audio_dir, basename+".mp3")
    return audio_file

def get_audio_length(audio_dir, basename, audio_length_cache):
    audio_file = get_audio_path(audio_dir, basename)
    if audio_file in audio_length_cache:
        audio_length = audio_length_cache[audio_file]
    else:
        with audioread.audio_open(audio_file) as f:
            audio_length = float(f.duration)
        audio_length_cache[audio_file] = audio_length
    return audio_length

def get_basename(eval_file, dataset_name):
    basename = os.path.splitext(os.path.basename(eval_file))[0]
    if dataset_name == "ours":
        if basename.split("_")[-2] == "non":
            basename = "non_laugh/" + basename
        else:
            basename = "laugh/" + basename
    return basename

def compute_frame_wise_metrics(eval_laughter, gt_laughter):
    global frame_wise_metrics
    metrics = precision_recall_fscore_support(gt_laughter, eval_laughter, zero_division=0)
    
    if len(metrics[3]) == 1: # only one class (all the values are the same)
        if gt_laughter[0] == 0:
            print("All TN")
            return
        elif gt_laughter[0] == 1:
            frame_wise_metrics["accuracy"].append(1.)
            frame_wise_metrics["precision"].append(1.)
            frame_wise_metrics["recall"].append(1.)
            frame_wise_metrics["f1"].append(1.)
        else:
            raise ValueError("Unknown gt_laughter[0]: {}".format(gt_laughter[0]))
    else:
        frame_wise_metrics["accuracy"].append((gt_laughter == eval_laughter).sum() / len(gt_laughter))
        frame_wise_metrics["precision"].append(metrics[0][1])
        frame_wise_metrics["recall"].append(metrics[1][1])
        frame_wise_metrics["f1"].append(metrics[2][1])

def compute_time_diff(eval_laughter_dict, gt_laugh_segment, audio_length):
    closest_values = sorted(eval_laughter_dict.values(), key=lambda x:abs(x["start_sec"]-gt_laugh_segment["start_sec"]))
    if len(closest_values) == 0:
        raise ValueError("No valid eval found")
    closest_value = closest_values[0]
    # # skip if not overlaping with gt
    # if closest_value["end_sec"] < gt_start_time or gt_end_time < closest_value["start_sec"]:
    #     continue
    eval_start_time = closest_value["start_sec"]
    eval_end_time   = closest_value["end_sec"]
    assert eval_start_time <= eval_end_time
    assert 0 <= eval_start_time and eval_end_time <= audio_length + 1., f"should {eval_start_time} <= {eval_end_time} <= {audio_length}"
    gt_start_time = gt_laugh_segment["start_sec"]
    gt_end_time   = gt_laugh_segment["end_sec"]

    return (gt_start_time - eval_start_time), (gt_end_time - eval_end_time)

def r(val):
    # return round(val, 3)
    return Decimal(val).quantize(Decimal('0.001'), ROUND_HALF_UP)

def main(dataset_name, neg_sample_scale=4):
    global frame_wise_metrics
    
    gt_dir = os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "gt")
    audio_dir = os.path.join(os.path.dirname(__file__), "datasets", dataset_name, "audio")
    model_dir = os.path.join(os.path.dirname(__file__), "models")

    audio_length_cache = {}
    cache_file = os.path.join(os.path.dirname(__file__), "cache", dataset_name+".json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            audio_length_cache = json.load(f)

    print(f"{input_sec} sec input window")
    print(f"Using {dataset_name} dataset")
    _debug_gallick_samples = set()
    _debug_ours_samples = set()
    # for model_name in os.listdir(model_dir):#["ours"]:
    # for model_name in ["ours"]:
    for model_name in ["Gillick2021_with_ourData"]:

        _variant_if_any = ""
        if model_name == "ours":
            _variant_if_any = ""#"2024-02-28_17-26-25LongMusicNeg_voi_misc_etc_fewDataArgu_jonatasgrosman_wav2vec2-large-xlsr-53-english_cleanUp3_oldProb_fe_fade"#"2024-02-08_11-29-32LongMusicNeg_voi_misc_etc_fewDataArgu_jonatasgrosman_wav2vec2-large-xlsr-53-english"
        if _variant_if_any:
            if not os.path.exists(os.path.join(model_dir, model_name, _variant_if_any)):
                raise ValueError(f"Variant {_variant_if_any} not found")
            print(f"[WARNING!] Using VARIANT: {_variant_if_any}")

        np.random.seed(42)
        random.seed(42)
        # if not os.path.isdir(os.path.join(model_dir, model_name)):
        if not os.path.isdir(os.path.join(model_dir, model_name, _variant_if_any)):
            continue
        eval_dir = os.path.join(model_dir, model_name, _variant_if_any, dataset_name)
        
        assert len(glob.glob(eval_dir+"/*.json")), f"No eval files found in {eval_dir}"
        gt_files = glob.glob(gt_dir+"/*.json")
        if len(gt_files) == 0:
            print("No gt/*.json found. Trying gt/*laugh/*.json")
            gt_files = glob.glob(gt_dir+"/*laugh/*.json")

        TP = 0; FP = 0; FN = 0; TN = 0
        detection_eval = []
        detection_gt = []
        tp_diff_start = []; tp_diff_end = []
        frame_wise_metrics = {"accuracy": [],
                              "precision": [],
                              "recall": [],
                              "f1": [], 
                             }

        prev_gt_laughters = 0
        # shuffle for random sampling gt
        random.shuffle(gt_files)
        for gt_file in gt_files:
            eval_file = os.path.join(eval_dir, os.path.basename(gt_file))
            basename = get_basename(eval_file, dataset_name)            

            if not os.path.exists(eval_file):
                eval_laughter_dict = {}
            else:
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_laughter_dict = json.load(f)
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_laughter_dict = json.load(f)
                gt_laughter_dict = {k:v for k,v in gt_laughter_dict.items() if not should_skip_sample(v)}
            
            if model_name == "ours":
                eval_laughter_dict = concat_close(eval_laughter_dict, 0.2)
                eval_laughter_dict = remove_short(eval_laughter_dict, 0.2)
            
            audio_length = get_audio_length(audio_dir, basename, audio_length_cache)
            hz = 100 # 100hz means devide 1 sec into 100 parts
            eval_laughter = np.zeros(int(audio_length*hz))
            gt_laughter = np.zeros(int(audio_length*hz))
            
            for eval_laugh_segment in eval_laughter_dict.values():
                start_time = int(eval_laugh_segment["start_sec"]*hz)
                end_time   = int(eval_laugh_segment["end_sec"]*hz)
                eval_laughter[start_time:end_time] = 1
            for gt_laugh_segment in gt_laughter_dict.values():
                start_time = int(gt_laugh_segment["start_sec"]*hz)
                end_time   = int(gt_laugh_segment["end_sec"]*hz)
                gt_laughter[start_time:end_time] = 1

            if dataset_name in datasets_by_dynamic_generation:
                for gt_laugh_segment in gt_laughter_dict.values():
                    gt_start_time = max(0, gt_laugh_segment["start_sec"] - 4)
                    gt_end_time   = min(gt_start_time + input_sec, audio_length)
                    if np.sum(eval_laughter[int(gt_start_time*hz):int(gt_end_time*hz)]) >= 1:
                        TP += 1
                        detection_eval.append(1); detection_gt.append(1)
                        compute_frame_wise_metrics(eval_laughter[int(gt_start_time*hz):int(gt_end_time*hz)], gt_laughter[int(gt_start_time*hz):int(gt_end_time*hz)])
                        tp_s, tp_e = compute_time_diff(eval_laughter_dict, gt_laugh_segment, audio_length)
                        tp_diff_start.append(tp_s); tp_diff_end.append(tp_e)
                    else:
                        FN += 1
                        detection_eval.append(0); detection_gt.append(1)
                for _ in range(int(len(gt_laughter_dict)*neg_sample_scale)):
                    start_time = np.random.uniform(0, audio_length-input_sec)
                    end_time   = start_time + input_sec
                    if np.sum(gt_laughter[int(start_time*hz):int(end_time*hz)]) >= 1: continue
                    if np.sum(eval_laughter[int(start_time*hz):int(end_time*hz)]) >= 1:
                        FP += 1
                        detection_eval.append(1); detection_gt.append(0)
                    else:
                        TN += 1
                        detection_eval.append(0); detection_gt.append(0)
            else:
                if np.sum(gt_laughter[:]) >= 1:
                    # has laughter in gt
                    if np.sum(eval_laughter[:]) >= 1:
                        TP += 1
                        detection_eval.append(1); detection_gt.append(1)
                        compute_frame_wise_metrics(eval_laughter, gt_laughter)
                        for gt_laugh_segment in gt_laughter_dict.values():
                            tp_s, tp_e = compute_time_diff(eval_laughter_dict, gt_laugh_segment, audio_length)
                            tp_diff_start.append(tp_s); tp_diff_end.append(tp_e)
                    else:
                        FN += 1
                        detection_eval.append(0); detection_gt.append(1)
                        # if model_name == "Gillick2021":
                        #     _debug_gallick_samples.add(basename)
                        # elif model_name == "ours":
                        #     if not basename in _debug_gallick_samples:
                        #         _debug_ours_samples.add(basename)
                else:
                    # no laughter in gt
                    if np.sum(eval_laughter[:]) >= 1:
                        FP += 1
                        detection_eval.append(1); detection_gt.append(0)
                        if model_name == "Gillick2021":
                            _debug_gallick_samples.add(basename)
                        elif model_name == "ours":
                            if not basename in _debug_gallick_samples:
                                _debug_ours_samples.add(basename)
                    else:
                        TN += 1
                        detection_eval.append(0); detection_gt.append(0)

        
        # if model_name == "ours" and _debug_ours_samples:
        #     print(_debug_gallick_samples, "\n")
        #     print(_debug_ours_samples, "\n")
        gt_laughter_count = TP + FN
        abs_tp_diff_start = list(map(abs, tp_diff_start))
        abs_tp_diff_end = list(map(abs, tp_diff_end))
        # print(f"""\
        # Found laughter count: {len(tp_diff_start)}/{gt_laughter_count} (=Recall: {len(tp_diff_start)/gt_laughter_count})
        
        # TP(笑いと予測し、GTも笑いだった): {TP}
        #     MSE start: {mean_squared_error([0]*len(tp_diff_start), tp_diff_start)}
        #     Max start diff: {max(abs_tp_diff_start)}[sec]            
        #     Average start diff: {mean(abs_tp_diff_start)}[sec]
        #     Stdev start diff: {stdev(tp_diff_start)}[sec] (-stdev～stdev: 68% of all)
        # FP(笑いと予測し、GTは笑いじゃなかった): {FP}    ←これが高いとノイズが多くなる
        # FN(笑いじゃないと予測し、GTは笑いだった): {FN}  ←これが高いのはそこまで問題ない。データセットの規模が小さくなるだけ
        # TN(笑いじゃないと予測し、GTも笑いじゃなかった): {TN}
        # Accuracy(TP,TNの多さ): {(TP+TN)/(TP+FP+FN+TN)}
        # Precision(笑いとの予測のうち、GTが実際に笑いか): {TP/(TP+FP)}    ←これを1にするタスクがしたい
        # Recall(笑いGTの中でどれくらい正解したか): {TP/(TP+FN)}
        # F1: {2*TP/(2*TP+FP+FN)}

        # Ave. frame-wise metrics:
        #     Accuracy: {mean(frame_wise_metrics["accuracy"])}
        #     Precision: {mean(frame_wise_metrics["precision"])}
        #     Recall: {mean(frame_wise_metrics["recall"])}
        #     F1: {mean(frame_wise_metrics["f1"])}
        # """)
        print("\n", model_name)
        # csv
        print(f"""\
{gt_laughter_count},{TP},{FP},{FN},{TN},\
{(TP+TN)/(TP+FP+FN+TN)},{TP/(TP+FP)},{TP/(TP+FN)},{2*TP/(2*TP+FP+FN)},\
{mean(frame_wise_metrics["accuracy"])},{mean(frame_wise_metrics["precision"])},{mean(frame_wise_metrics["recall"])},{mean(frame_wise_metrics["f1"])},\
{mean_squared_error([0]*len(tp_diff_start), tp_diff_start)},{max(abs_tp_diff_start)},{mean(abs_tp_diff_start)},{stdev(tp_diff_start)},\
{mean_squared_error([0]*len(tp_diff_end),   tp_diff_end)  },{max(abs_tp_diff_end)  },{mean(abs_tp_diff_end)  },{stdev(tp_diff_end)  },\
{TP+FN},{FP+TN}
        """)

        # confidence intervals # alpha=5% (95% confidence)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrices = [accuracy_score, precision_score, recall_score, f1_score]
        print("Detection metrices:")
        for metric in metrices:
            res = evaluate_with_conf_int(np.array(detection_eval), metric, np.array(detection_gt), conditions=None, num_bootstraps=len(detection_gt), alpha=5)
            # print(res)
            print(f"    {metric.__name__}: {r(res[0])}±{(r(res[1][1]-res[1][0])/2)}")
        metrices = ["accuracy", "precision", "recall", "f1"]
        print("Segmentation metrices:")
        for metric in metrices:
            res = evaluate_with_conf_int(np.array(frame_wise_metrics[metric]), mean, None, conditions=None, num_bootstraps=len(frame_wise_metrics[metric]), alpha=5)
            # print(r(res)
            print(f"    {metric}: {r(res[0])}±{(r(res[1][1]-res[1][0])/2)}")
        from sklearn.metrics import mean_absolute_error
        mae_start = evaluate_with_conf_int(np.array(tp_diff_start), mean_absolute_error, np.array([0]*len(tp_diff_start)), conditions=None, num_bootstraps=len(tp_diff_start), alpha=5)
        print(f"Start MAE: {r(mae_start[0])}±{(r(mae_start[1][1]-mae_start[1][0])/2)}")
        mae_end = evaluate_with_conf_int(np.array(tp_diff_end), mean_absolute_error, np.array([0]*len(tp_diff_end)), conditions=None, num_bootstraps=len(tp_diff_end), alpha=5)
        print(f"End MAE: {r(mae_end[0])}±{(r(mae_end[1][1]-mae_end[1][0])/2)}")
            

if __name__ == '__main__':
    # main("Petridis2013", 2.6)
    # main("McCowan2005", 1.27)
    # # main("Liu2022")
    # main("Gillick2021")
    main("ours")