import argparse

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def save_log(date, id, comment):
    import csv, os
    if os.path.exists('./out/train_log.csv'):
        with open('./out/train_log.csv', 'r', encoding="utf-8", newline='') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

            if rows and rows[-1][1].lower() == id.lower():
                raise Exception('ID is same as previous train.')
    
    with open('./out/train_log.csv', 'a+', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date, id, comment])

def git_backup(id):
    import git, os
    repo = git.Repo("../../")
    # It is strange but when using VS Code, all changed or added files are added automatically
    changedFiles = [item.a_path for item in repo.index.diff(None)]

    # warn if a file is too large
    # this is basically for jupyter notebook with media output like audio
    thre_size = 1048576 # 1MB
    for file in changedFiles:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", file)
        if os.path.exists(file_path) and os.path.isfile(file_path) and os.path.getsize(file_path) > thre_size:
            raise ValueError(f"Modified file {file} size may be too large.")

    repo.git.add("-A")
    if changedFiles:
        repo.git.commit("-m", "🤖"+id)
    else:
        repo.git.commit("-m", "🤖"+id+"(empty)", "--allow-empty")
    print("commit as", id)
"""

def main(input_sec, id, comment, checkpoint, debug):
    import datetime
    import torch
    from transformers.trainer_utils import set_seed
    seed = 42 if not debug else datetime.datetime.now().second
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    date = str(datetime.datetime.now()).split(".")[0].replace(" ","_").replace(":","-")
    # if not debug:
    #     save_log(date, id, comment)
    #     git_backup(id)

    from transformers import TrainingArguments, Trainer, TrainerCallback
    from transformers import AdamW, get_linear_schedule_with_warmup
    import evaluate
    # from accelerate import Accelerator
    
    from model import Model
    from Data import Dataset, CustomDataCollator
    from copy import deepcopy

    lang = "en"
    if lang == "en":
        audio_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    
    audio_preprocessor_name = audio_model_name

    #how many gpus are available
    print("GPUs available: ", torch.cuda.device_count())
    # accelerator = Accelerator()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = accelerator.device

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    

    batch_size = 5

    dataset = Dataset(lang, 10 if debug else None, False)
    data_collator = CustomDataCollator(input_sec, audio_preprocessor_name, debug)
    train_dataset = dataset.encoded_dataset["train"]; eval_dataset = dataset.encoded_dataset["validation"]

    model = Model(audio_model_name, device, data_collator.sr)

    out_dir = "out/"+date+id
    lr = 5e-6 
    
    logging_steps = 100
    num_train_epochs= 5
    max_steps=num_train_epochs*2500//batch_size
    eval_steps = 250
    training_args = TrainingArguments(
        output_dir=out_dir,
        # resume_from_checkpoint=checkpoint,
        do_train=True,
        do_eval=True,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps, # important when streaming. Overwrite num_train_epochs
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers = 0,
        dataloader_persistent_workers = False,
        dataloader_pin_memory = True,
        # save_safetensors = False, # when True, train from checkpoint fails
        # gradient_accumulation_steps=15//batch_size, # nearly equal to this * batchsize, deafault to 15 for now
        dataloader_drop_last=True,
        weight_decay=0.01,
        evaluation_strategy="steps",#"epoch",#"steps""no"
        eval_steps=1 if debug else eval_steps, #if evaluation_strategy==step
        # label_smoothing_factor=.05, # seems useful but need more understand... https://github.com/huggingface/transformers/blob/b338414e614a30af5f940269484ef15bf716d078/src/transformers/trainer_pt_utils.py#L472
        disable_tqdm=False,
        logging_steps=logging_steps,
        ignore_data_skip=True,
        push_to_hub=False,
        log_level="debug",#"error",
        logging_dir=out_dir+"/logs",
        save_strategy="steps",#"epoch", #steps",#"epoch",#, #checkpointを保存する頻度
        save_steps=eval_steps, #if save_strategy=="steps"
        # load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1, #total amount of checkpoints. Deletes the older checkpoints in output_dir.
        fp16=True,
        lr_scheduler_type="linear", #linear, cosine,"constant"
        remove_unused_columns=False,
        seed=seed,
    )


    # 評価指標の定義
    def compute_metrics(pred):
        preds, labels = pred
        if preds.ndim == 3:
            preds = preds[:,0,:].squeeze()
        
        preds = torch.sigmoid(torch.tensor(preds, dtype=torch.float32))

        ### TP detection only ###
        preds = (preds>=.5).int()
        labels = torch.tensor(labels).type(torch.int)
        preds = preds.max(dim=1)[0] # get if each batch has at least one 1
        labels = labels.max(dim=1)[0]

        ### normal
        # preds = (preds>=.5).int().flatten()
        # labels = torch.tensor(labels).type(torch.int).flatten()

        metrics = clf_metrics.compute(predictions=preds, references=labels)
        return metrics

    # model, dataset = accelerator.prepare(model, dataset)
    class TrainLogCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if int(state.epoch) % 150 != 0:
                return control
            if True:#control.should_evaluate:
                control_copy = deepcopy(control)
                metrics = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                trainer.save_metrics("train", metrics)
                return control_copy

    params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = AdamW(params, lr=lr, correct_bias = True)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
            num_warmup_steps=0, num_training_steps=max_steps)

    # トレーニング
    trainer = Trainer(
        model=model, args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=[optimizer, scheduler],
        # optimizers=[optimizer, None],
        # callbacks=[TrainLogCallback],
    )

    import warnings
    warnings.simplefilter('ignore', UserWarning)

    # Training
    if training_args.do_train:
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        print("checkpoint", checkpoint)
        trainer.add_callback(TrainLogCallback(trainer))
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()

    if not args.debug and not args.id:
        raise ValueError("--id is required unless debug")
    if args.debug and not args.id:
        args.id = "test"

    main(7, args.id, args.comment, args.checkpoint, args.debug)