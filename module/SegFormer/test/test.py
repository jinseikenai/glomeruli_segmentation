from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from SegFormer.common.ResizedGlomerularDataset import ResizedGlomerularDataset
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import SegFormer.common.Transforms as myTransforms
# from datasets import load_metric
from SegFormer.common import mean_iou as my_mean_iou
from argparse import ArgumentParser
import csv
from accelerate import Accelerator
import ast
import glob

font = ImageFont.truetype(font="NotoSansCJK-Bold.ttc", size=62)


def glomerular_palette():
    # overlay させるために色を少し強調しておく
    return [[0, 0, 0], [120, 120, 120], [250, 47, 0],  [0, 220, 58], [43, 90, 250], [255, 255, 100]]
    # return [[0, 0, 0], [120, 120, 120], [213, 47, 0],  [0, 180, 58], [43, 90, 233], [255, 255, 128]]
    # return [[0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [255, 255, 128]]


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits_tensor, labels = eval_pred
        # logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        # labels は numpy arrays になっている
        # compute の I/Fでは tensor を受け付けることが出来るが内部的に numpy に変換される。
        pred_labels = logits_tensor.detach().cpu().numpy()
        '''
        metrics = metric.compute(predictions=pred_labels, references=labels,
                                 num_labels=args.num_labels,
                                 ignore_index=255,
                                 reduce_labels=feature_extractor.reduce_labels)
        '''
        metrics = metric.mean_iou(results=pred_labels, gt_seg_maps=labels,
                                  num_labels=args.num_labels,
                                  ignore_index=255,
                                  reduce_labels=feature_extractor.reduce_labels)

        pred_pixel = {k: 0 for k in range(args.num_labels)}
        label_pixel = {k: 0 for k in range(args.num_labels)}
        pred_label = pred_labels.squeeze()
        for k, pp in pred_pixel.items():
            # pred_pixel を tensor のままにした場合は item() で値を取り出す必要がある。
            # metric.compute の内部では numpy を使うので最初から numpy に変換しておく
            pred_pixel[k] = (pred_label == k).sum()
            label_pixel[k] = (labels == k).sum()
        # ignore_index=0,
        '''
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        '''
        # 画像を生成して保存する
        if args.save_image:
            save_image(pred_label, np.squeeze(labels))
        return metrics, label_pixel, pred_pixel


def collate_fn(batch):
    batch_list = [list(x.values()) for x in batch]
    images = [x[0] for x in batch_list]
    labels = [x[1] for x in batch_list]
    images = torch.stack(images)
    # labels の大きさはそれぞれ異なるので torch.stack 出来ない
    # targets = torch.stack(targets)
    return images, labels


# ここでは簡易的に mpp を 0.228 に固定している
slide_info_mppx = 0.228
scale_bar_length = round(100.0 / slide_info_mppx)


def save_image(pred_seg: np.array, gt_seg: np.array):
    """

    :param pred_seg: np.array
    :param gt_seg: np.array
    :return:
    """
    pred_img = Image.fromarray(np.uint8(pred_seg), mode='L')
    save_pred_image_path = os.path.join(report_root_path, 'seg', specimen_id)
    if not os.path.isdir(save_pred_image_path):
        os.makedirs(save_pred_image_path)
    save_file_path = os.path.join(save_pred_image_path, file_name)
    pred_img.save(save_file_path)

    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    color_gt_seg = np.zeros((gt_seg.shape[0], gt_seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    palette = np.array(glomerular_palette())
    for label, color in enumerate(palette):
        color_seg[pred_seg == label, :] = color
        color_gt_seg[gt_seg == label, :] = color

    save_file_dir = os.path.join(report_root_path, specimen_id)
    if not os.path.isdir(save_file_dir):
        os.makedirs(save_file_dir)
    save_file_path = os.path.join(save_file_dir, file_name)
    org_img = Image.open(image_file_name)
    org_img = org_img.convert(mode='RGBA')
    seg_img = Image.fromarray(color_seg)
    seg_img = seg_img.convert(mode='RGBA')
    if org_img.size != seg_img.size:
        print(f'image size not match: {save_file_path}')
    seg_img = Image.blend(org_img, seg_img, 0.7)
    # seg_img = Image.alpha_composite(org_img, seg_img)
    # seg_img.show()
    gt_img = Image.fromarray(color_gt_seg)
    gt_img = gt_img.convert(mode='RGBA')
    gt_img = Image.blend(org_img, gt_img, 0.7)
    # gt_img = Image.alpha_composite(org_img, gt_img)
    # gt_img.show()
    concat_img = Image.new(mode='RGBA', size=(org_img.width + seg_img.width + gt_img.width, org_img.height))
    ''''''
    # if scale_bar:
    # 100μm のスケールバーを挿入する
    width, height = org_img.size
    draw = ImageDraw.Draw(org_img)
    draw.line((30, height-30, scale_bar_length+30, height-30), fill='black', width=16)
    draw.text((int(scale_bar_length / 2) - 70, height-114), text='100 μm', fill='black',
              font=font)
    ''''''
    concat_img.paste(org_img, (0, 0))
    concat_img.paste(seg_img, (org_img.width, 0))
    concat_img.paste(gt_img, (org_img.width+seg_img.width, 0))
    # concat_img.show()
    concat_img.save(save_file_path)


def search_best_checkpoint(model_base_path: str):
    best_iou = 0.0
    best_epoch = 0
    last_epoch = 0
    with open(os.path.join(model_base_path, 'log.txt'), mode='r') as log:
        for line in log.readlines():
            if 'eval_mean_iou' in line:
                line = line[line.find('{'): line.find('}')+1]
                d = ast.literal_eval(line)
                eval_mean_iou = float(d['eval_mean_iou'])
                if best_iou < eval_mean_iou:
                    best_iou = eval_mean_iou
                    best_epoch = int(d['epoch'])
                last_epoch = int(d['epoch'])
    checkpoints = glob.glob(os.path.join(model_base_path, 'checkpoint-*'))
    assert len(checkpoints) > 0, 'checkpoints does not found.'
    cps = [int(os.path.basename(x).replace('checkpoint-', '')) for x in checkpoints]
    cps.sort()
    if best_epoch == last_epoch:
        best_checkpoint = cps[-1]
    else:
        best_checkpoint = cps[-2]
    return f'checkpoint-{best_checkpoint}'


if __name__ == '__main__':
    parser = ArgumentParser(description='segformer')
    parser.add_argument('--num_labels', help="set number of label", type=int,
                        default=5)
    parser.add_argument('--batch_size', help="set batch size", type=int,
                        default=2)
    parser.add_argument('--fold', help="set number of fold", type=int,
                        required=True)
    parser.add_argument('--target_site', help="set test target site", type=str,
                        choices=["01_Todai", "02_Kitano"], required=True)
    parser.add_argument('--model_site', help="set model site", type=str,
                        choices=["01_Todai", "02_Kitano"], required=True)
    parser.add_argument('--data_date', help="set training data date", type=str,
                        required=True)
    parser.add_argument('--pretrained_model', help="set pretrained model", type=str,
                        default="segformer/20220804_b4")
    parser.add_argument('--checkpoint', help="set continue checkpoint", type=str,
                        required=True, default='')
    parser.add_argument('--save_image', help="set 1 if save image", type=int,
                        default=0)
    parser.add_argument('--report_root_path', help="set report root path", type=str,
                        required=True)
    parser.add_argument('--data_root', help="set data_root", type=str,
                        required=True)
    parser.add_argument('--model_base_path', help="set model base path", type=str,
                        required=True)
    parser.add_argument('--detected_mode', help="set 1 for segmentation to detected regions", type=int,
                        default=0)
    args = parser.parse_args()

    # feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
    feature_extractor = SegformerFeatureExtractor(reduce_labels=False)
    if args.checkpoint == '':
        # checkpoint が指定されていない場合は最良のチェックポイントの番号を探す。
        model_base_path = os.path.join(args.model_base_path, f'{args.model_site}/{args.pretrained_model}/fold{args.fold}')
        checkpoint = search_best_checkpoint(model_base_path)
    else:
        checkpoint = args.checkpoint
    model_path = os.path.join(args.model_base_path, f'{args.model_site}/{args.pretrained_model}/fold{args.fold}/{checkpoint}')
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)

    data_source = os.path.join(args.data_root, args.target_site, args.data_date)
    # test の場合も ToTensor() だけは必要
    testDatasetTransforms = myTransforms.Compose([
        myTransforms.ToTensor(),
    ])
    test_ds = ResizedGlomerularDataset(root_dir=data_source, rgb_subdir='rgb', label_subdir='label/gtcs',
                                       feature_extractor=feature_extractor,
                                       transforms=testDatasetTransforms,
                                       mode='test', fold=args.fold,
                                       detected_mode=args.detected_mode)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn,
                                 num_workers=0, pin_memory=False)
    accelerator = Accelerator(fp16=False)
    test_dataloader, test_ds, model = accelerator.prepare(
        test_dataloader, test_ds, model
    )

    # metric = load_metric("mean_iou")
    metric = my_mean_iou
    # ここまでデータ準備

    # ここからレポートファイル準備
    report_root_path = os.path.join(args.report_root_path, args.target_site, args.model_site, args.data_date,
                                    args.pretrained_model, f'fold{args.fold}')
    if not os.path.isdir(report_root_path):
        os.makedirs(report_root_path)

    image_file_list = test_ds.images
    data_len = len(image_file_list)
    with open(os.path.join(report_root_path, 'pred_summary_pixel.csv'), mode='w') as summary_pixel:
        writer = csv.writer(summary_pixel)
        writer.writerow(['specimen_id', 'filename', 'background', 'glomerulus', 'tuft', 'crescent', 'sclerosis', 'mIoU'])
        metrics_sum = {}
        idx = 0

        for i, data in enumerate(test_dataloader):
            # images, gt_seg = next(iter(test_dataloader))
            images, gt_seg = data

            outputs = model(images)
            # shape (batch_size, num_labels, height/4, width/4)
            logits = outputs.logits
            # test 時は gt_seg のイメージサイズがバラバラなので一つずつ評価する
            for pred, gt in zip(logits, gt_seg):
                image_file_name = image_file_list[idx]
                split_image_file_name = image_file_name.split('/')
                specimen_id = split_image_file_name[-2]
                file_name = split_image_file_name[-1]
                # compute_metrics はバッチ前提なので pred と gt ともにバッチ軸を追加する
                pred = pred.unsqueeze(dim=0)
                metrics, label_pixel, pred_pixel = compute_metrics((pred, gt[np.newaxis, :, :]))
                for key, value in metrics.items():
                    if key not in metrics_sum:
                        metrics_sum[key] = value
                    else:
                        metrics_sum[key] += value
                idx += 1
                if idx % 10 == 0:
                    print(f'{idx}/{data_len}')
                pixels_base = metrics['total_area_pred_label']
                pixels = [pixels_base[0],
                          pixels_base[1] + pixels_base[2] + pixels_base[3] + pixels_base[4],
                          pixels_base[2], pixels_base[3], pixels_base[4]]
                writer.writerow([specimen_id, file_name] + pixels + [metrics['mean_iou']])

    overall_iou = np.zeros(shape=args.num_labels, dtype=metrics_sum['total_area_intersect'].dtype)
    overall_acc = np.zeros(shape=args.num_labels, dtype=metrics_sum['total_area_intersect'].dtype)
    for key, value in metrics_sum.items():
        if key in ['total_area_intersect', 'total_area_union', 'total_area_label']:
            if key == 'total_area_union':
                overall_iou = metrics_sum['total_area_intersect'] / metrics_sum['total_area_union']
            elif key == 'total_area_label':
                overall_acc = metrics_sum['total_area_intersect'] / metrics_sum['total_area_label']
        else:
            metrics_sum[key] /= data_len
    metrics_sum["overall_iou"] = overall_iou
    metrics_sum["overall_acc"] = overall_acc
    metrics_sum["overall_mean_acc"] = np.nanmean(metrics_sum['overall_acc'])
    metrics_sum["overall_mean_iou"] = np.nanmean(metrics_sum['overall_iou'])
    del metrics_sum['per_category_iou'], metrics_sum['per_category_accuracy'],\
        metrics_sum['total_area_intersect'], metrics_sum['total_area_union'], \
        metrics_sum['total_area_label'], metrics_sum['overall_accuracy'], metrics_sum['total_area_pred_label']
    for key, value in metrics_sum.items():
        if type(value) is np.ndarray:
            metrics_sum[key] = value.tolist()
    print(metrics_sum)
    with open(os.path.join(report_root_path, 'summary_report.csv'), mode='w') as overall:
        writer = csv.writer(overall)
        writer.writerow(['metric', 'value', 'background', 'glomerulus', 'tuft', 'crescent', 'sclerosis'])
        for key, value in metrics_sum.items():
            if isinstance(value, list):
                writer.writerow([key, ''] + value)
            else:
                writer.writerow([key, value])

print('end of test.')
