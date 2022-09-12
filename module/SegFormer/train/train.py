from transformers import SegformerModel, SegformerConfig
from transformers import SegformerFeatureExtractor, SegformerModel
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
from SegFormer.common.GlomerularDataset import GlomerularDataset
from SegFormer.common.PreprocessedGlomerularDataset import PreprocessedGlomerularDataset
from SegFormer.common.ResizedGlomerularDataset import ResizedGlomerularDataset
import os
import json
from huggingface_hub import cached_download, hf_hub_url
from datasets import load_metric
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
from accelerate import Accelerator
import math
from transformers import TrainingArguments
from transformers import Trainer, TrainerCallback
import logging
import numpy as np
import SegFormer.common.Transforms as myTransforms
import sys


class LoggerLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        control.should_log = False
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)  # using your custom logger


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        # compute の I/Fでは tensor を受け付けることが出来るが内部的に numpy に変換される。
        # labels は numpy arrays になっている
        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(predictions=pred_labels, references=labels,
                                 num_labels=args.num_labels,
                                 ignore_index=255,
                                 reduce_labels=feature_extractor.reduce_labels)
                                 # ignore_index=0,
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics


def main(max_epoch):
    torch.backends.cudnn.benchmark = True
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        train_dataset.set_mode('train')
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            # pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            # labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            # loss_cpu = loss.detach().cpu().numpy()

            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()

            # let's print loss and metrics every 100 batches
            # if (idx+1) % epoch_idx == 0:
        # reports metrics at epoch end.
        train_dataset.set_mode('val')
        batch = next(iter(train_dataloader))
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        # evaluate
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            metrics = metric.compute(num_labels=args.num_labels,
                                     ignore_index=255,
                                     reduce_labels=feature_extractor.reduce_labels,  # we've already reduced the labels before)
                                     )

            print("Loss:", loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segformer')
    parser.add_argument('--num_labels', help="set number of label", type=int,
                        default=5)
    parser.add_argument('--batch_size', help="set batch size", type=int,
                        default=2)
    parser.add_argument('--dl_num_workers', help="set dataloader num works", type=int,
                        default=2)
    parser.add_argument('--max_epoch', help="set number of max epochs", type=int,
                        default=1000)
    parser.add_argument('--fold', help="set number of fold", type=int,
                        default=1)
    parser.add_argument('--site', help="set training target site", type=str,
                        choices=["01_Todai", "02_Kitano"], required=True)
    parser.add_argument('--data_date', help="set training data date", type=str,
                        required=True)
    parser.add_argument('--data_root', help="set training data root dir", type=str,
                        required=True)
    parser.add_argument('--model_root', help="set dir for output models", type=str,
                        required=True)
    parser.add_argument('--output_dir', help="set sub path to output dir", type=str,
                        default="20220720")
    parser.add_argument('--pretrained_model', help="set pretrained model", type=str,
                        default="nvidia/mit-b0")
    parser.add_argument('--lr', help="set max lr", type=float,
                        default=0.00006)
    parser.add_argument('--save_interval', help="set save interval epochs", type=int,
                        default=20)
    parser.add_argument('--accumulation_steps', help="set gradient_accumulation_steps", type=int,
                        default=1)
    parser.add_argument('--checkpoint', help="set continue checkpoint", type=str,
                        default="")
    args = parser.parse_args()

    # datasource = model_root + 'ADE20k_dataset/ADE20K_2016_07_26'
    # datasource = model_root + 'ADE20k_toy_dataset'
    data_source = os.path.join(data_root, args.site, args.data_date)
    # pretrained = model_root + 'pretrained/segformer.b1.512x512.ade.160k.pth'

    configuration = SegformerConfig()
    model = SegformerModel(configuration)
    configuration = model.config

    # dataset = load_dataset("huggingface/cats-image")
    # dataset = load_dataset(datasource)
    # image = dataset["train"][0]['image']

    # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
    # feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

    feature_extractor = SegformerFeatureExtractor(reduce_labels=False)

    #compose the data with transforms
    trainDatasetTransforms = myTransforms.Compose([
        # myTr    lr = 0.00006ansforms.Normalize(mean=data['mean'], std=data['std']),
        # myTransforms.Scale(1024, 512),
        # myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(64),
        myTransforms.RandomFlip(),
        myTransforms.RandomVerticalFlip(),
        myTransforms.RandomBlurringAndSharpning(),
        myTransforms.RandomContrast(),
        # myTransforms.RandomCrop(64).
        myTransforms.ToTensor(),
    ])

    valDatasetTransforms = myTransforms.Compose([
        myTransforms.ToTensor(),
    ])

    # train_dataset = GlomerularDataset(root_dir=data_source, feature_extractor=feature_extractor)
    train_ds = ResizedGlomerularDataset(root_dir=data_source, rgb_subdir='rgb', label_subdir='label/gtcs',
                                        feature_extractor=feature_extractor,
                                        transforms=trainDatasetTransforms,
                                        mode='train', fold=args.fold)
    val_ds = ResizedGlomerularDataset(root_dir=data_source, rgb_subdir='rgb', label_subdir='label/gtcs',
                                      feature_extractor=feature_extractor,
                                      transforms=valDatasetTransforms,
                                      mode='val', fold=args.fold)
    # valid_dataset = PickledGlomerularDataset(root_dir=data_source, feature_extractor=feature_extractor, train=False)

    print(f"Number of training examples: {len(train_ds)}")
    print(f"Number of validation examples: {len(val_ds)}")

    # Let's verify a random example:
    for idx in range(0, 2):
        encoded_inputs = train_ds[idx]
        print(f'image shape[{idx}]: {encoded_inputs["pixel_values"].shape}, label shape: {encoded_inputs["labels"].shape}')
        # print(f'labels : {encoded_inputs["labels"].unique()}')
        print(f'labels[{idx}] : {np.unique(encoded_inputs["labels"])}')

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                               num_workers=args.dl_num_works, pin_memory=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # load id2label mapping from a JSON on the hub
    # repo_id = "datasets/huggingface/label-files"
    # filename = "ade20k-id2label.json"
    # id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    # id2label = {int(k): v for k, v in id2label.items()}
    # label2id = {v: k for k, v in id2label.items()}

    # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512",
    model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_model,
                                                             num_labels=args.num_labels)
    #                                                          num_labels=150,
    #                                                          id2label=id2label,
    #                                                          label2id=label2id)
    epoch_steps = math.ceil(len(train_ds) / args.batch_size)
    training_args = TrainingArguments(
        output_dir=os.path.join(model_root, args.site, f'{args.output_dir}/fold{args.fold}'),
        learning_rate=args.lr,
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        # best と last の 2つのモデルを保存する
        save_total_limit=2,
        evaluation_strategy="steps",  # "no", "epoch", "steps"
        save_strategy="steps",  # "no", "epoch", "steps"
        # save_interval epoch 毎に 評価を行う。最初の保存 epoch は warmup に使う
        warmup_steps=epoch_steps*args.save_interval,
        save_steps=epoch_steps*args.save_interval,
        eval_steps=epoch_steps*args.save_interval,
        logging_steps=epoch_steps,
        gradient_accumulation_steps=args.accumulation_steps,
        eval_accumulation_steps=1,
        # best model 判定を mean_iou で行う。mean_iou を用いる場合は greater_is_better=Trueを設定する必要がある。
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        load_best_model_at_end=True,
        # push_to_hub=True,
        # hub_model_id="nvidia/mit-b0",
        # hub_strategy="every_save",
        fp16=True,
        fp16_opt_level='O1',
        hub_private_repo=False,
        dataloader_num_workers=args.dl_num_workers,
    )
    # define metric
    metric = load_metric("mean_iou")
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # move model to GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    accelerator = Accelerator(fp16=True)
    # device = Accelerator.device
    # model.to(device)
    train_ds, val_ds, model, optimizer = accelerator.prepare(
        train_ds, val_ds, model, optimizer
    )
    model.train()

    # file logger の準備
    log_file_dir = os.path.join(model_root, args.site, f'{args.output_dir}/fold{args.fold}')
    if not os.path.isdir(log_file_dir):
        os.makedirs(log_file_dir)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(filename=os.path.join(log_file_dir, 'log.txt'),
                                mode='a'),
            # logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('train')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[LoggerLogCallback],
    )
    # train_dataloader.num_workers = args.dl_num_works
    # eval_dataloader.num_workers = args.dl_num_works
    # main(max_epoch=args.max_epoch)
    if args.checkpoint == '':
        trainer.train()
    else:
        trainer.train(args.checkpoint)

    # model.save_pretrained(os.path.join(model_root, f'trained_model_fold{args.fold}'))

    print('end of process.')
