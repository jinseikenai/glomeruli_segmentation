# Pathohistological Glomerular Segmentation
This repository will introduce an example of pathohistological glomerular segmentation on a whole slide image.  
We uses the [ESPNet](https://arxiv.org/abs/1803.06815) method and the [Glomerular detection](https://github.com/jinseikenai/glomeruli_detection) for its implementation.
## Contents
1. [Requirements: hardware](#requirements-hardware)
2. [Requirements: software](#requirements-software)
3. [Basic installation](#basic-isntallation)
4. [Quick Start](#quick-start)
5. [Data Preparation](#data-preparation)
6. [Train](#train)
7. [Test](#test)

## Requirements: hardware
### Options
GPUs (e.g., Titan, P100, V100, ...) with at least 3G of memory suffices
## Requirements: software
### Requirements
- Docker (e.g., version 18.09.2, build 6247962)
### Options
- NVIDIA Docker (e.g., version 1.0.1), Docker Engine Utility for NVIDIA GPUs (e.g., version 2.0.2), NVIDIA Container Runtime for Docker (e.g., version 2.1.0), or NVIDIA Container Toolkit (e.g., version 2.2.1)
## Basic installation
### 1. Clone this repository

```bash
git clone --recursive https://github.com/jinseikenai/glomeruli_segmentation
```

### 2. Build docker image

```bash
docker build -t glomerular-espnet-pytorch:latest -f docker/gpu.dockerfile .
```

### 3. Run docker container

```bash
# If you do not use GPUs
docker run --shm-size 12G --rm -ti -v {mount directory at host}:{mount directory at container} glomerular-espnet-pytorch:latest /bin/bash
```
```bash
# If you use NVIDIA Docker
nvidia-docker run --shm-size 12G --rm -ti -v {mount directory at host}:{mount directory at container} glomerular-espnet-pytorch:latest /bin/bash
```
```bash
# If you use NVIDIA GPUs
docker run --runtime=nvidia --shm-size 12G --rm -ti -v {mount directory at host}:{mount directory at container} glomerular-espnet-pytorch:latest /bin/bash
```
```bash
# If you use NVIDIA GPUs
docker run --gpus 0 --shm-size 12G --rm -ti -v {mount directory at host}:{mount directory at container} glomerular-espnet-pytorch:latest /bin/bash
```

Everything below should be implemented in this container.
## Quick Start

  * [Quick Start Guide](https://github.com/jinseikenai/glomeruli_segmentation/blob/master/example/README.md) for getting Started to glomerular segmentation with our [models](https://github.com/jinseikenai/glomeruli_segmentation/blob/master/models) and [sample WSI](https://github.com/jinseikenai/glomeruli_segmentation/tree/master/example/data/02_PAS/H17-02419).  
## Data Preparation
See the directory "examples/" in this repository.  
If you want to evaluate the performance of segmentation, options below are necessary to prepare.  
```bash
    +-- <Parent directory>
        +-- opt_pas_test_list.txt: a file of target list. In the file, ndpi filenames without extension are written.
        |
        +-- data
            +-- 02_PAS
            |   +-- <Patient number>
            |       +-- annotations (option)
            |       |    +-- filename.xml: a ground truth file of glomerular detection
            |       +-- glomerular_wsi.ndpi: an original whole slide image. ndpi.
            +-- label (option)
            |   +-- all
            |       +-- <Patient number>
            |           +-- glomerular_label_name.PNG: single glomerular label images. This should be created from glomerular_image_name.json below.
            |
            +-- seg_annotation (option)
                +-- <Patient number>
                    +-- glomerular_image_name.PNG: single glomerular images
                    +-- glomerular_image_name.json: single glomerular label data
```

## Train
### Data Preparation
```bash
    +-- <Parent directory>
        +-- train: training data
        |   +-- rgb
        |   |   +-- glomerular_image_name.PNG: original single glomerular images
        |   +-- label
        |   |   +-- glomerular_label_name.PNG: single glomerular label images. This should be created from glomerular_image_name.json below. The image name must be the same as the original single glomerular image.
        +-- val: validation data
            +-- rgb
            |   +-- glomerular_image_name.PNG: original single glomerular images
            +-- label
                +-- glomerular_label_name.PNG: single glomerular label images. This should be created from glomerular_image_name.json below. The image name must be the same as the original single glomerular image.
        
```

```bash
# Create dataset list. This program output train.txt and val.txt in the "--data_dir".
python /opt/ESPNet/train/create_dataset_txt.py \
    --data_dir={Required. Set path to the parent data directory.}
```
This program outputs files below in "--data_dir".

```bash
+-- <data_dir>
    +-- train.txt: absolute path to the training rgb and label images are written.
    +-- val.txt: absolute path to the validation rgb and label images are written.
```

```bash
# Start encoder training from scratch
python /opt/ESPNet/train/main.py \
    --data_dir={Required. Set path to the parent data directory.} \
    --classes={Required. Set the number of classes including background} \
    --scaleIn=8 \
    --p=2 \
    --q=8 \
    --gpu_id={Required. Set GPU ID} \
    --savedir={Required. Set path to the output directory.} \
    --max_epochs={Required. Set the number of epochs.}
```

This program outputs files below in "savedir".

```bash
+-- <savedir>_enc_2_8
    +-- acc_{num of epoch}.txt:  accuracy and mIoU
    +-- model_{num of epoch}.pth: trained snapshot
    +-- model.png: depicted network architecture
    +-- trainValLog.txt: loss and  mIoU for trainig and validation
    +-- mean_std.txt: Mean and Standard deviation of RGB values of all training images.
```

```bash
# Start decoder training with the use of the encoder trained model
python /opt/ESPNet/train/main.py \
    --data_dir={Required. Set path to the parent data directory.} \
    --classes={Required. Set the number of classes including background} \
    --scaleIn=1 \
    --p=2 \
    --q=8 \
    --decoder=True \
    --pretrained={Required. Set path to the best encoder trained model.} \
    --gpu_id={Required. Set GPU ID} \
    --savedir={Required. Set path to the output directory.} \
    --max_epochs={Required. Set the number of epochs.|

```
This program outputs files below in "savedir".

```bash
+-- <savedir>_dec_2_8
    +-- acc_{num of epoch}.txt:  accuracy and IoU of each classes
    +-- model_{num of epoch}.pth: trained snapshot
    +-- model.png: depicted network architecture
    +-- trainValLog.txt: loss and  mIoU for trainig and validation
    +-- mean_std.txt: Mean and Standard deviation of RGB values of all training images.
```

## Test
### 1. Glomerular detection
  * Clone the repository of [Glomerular detection](https://github.com/jinseikenai/glomeruli_segmentation/blob/master/glomerular_segmentation.md). Implement 1. Glomeruli Detection and 2. Merging Overlapping Regions.
  * Use the [trained model](https://ai-health.m.u-tokyo.ac.jp/labweb/dl/renal_ai/segmentation/finetuned_frcnn.tar.gz)
  * Make cropped images for segmentaion
  ```bash
  python /opt/glomeruli_detection/make_seg_data.py \
      --staining=OPT_PAS \
      --target_list={Required. Set path to the file of target list. In the file, ndpi filenames without extension are written.} \
      --merged_detection_result_csv={Required. Set path to the merged detection result (csv file format)} \
      --wsi_dir={Required. Set path to the parent directory of whole slide images} \
      --segmentation_gt_json_dir={Option. Set path to parent directory of the segmentation grount truth json files. If you do not have the ground truth, no need to set this argument.} \
      --object_detection_gt_xml_dir={Option. Set path to the parent directory of the segmentation object detection xml files. If you do not have the ground truth, no need to set this argument.} \
      --segmentation_gt_png_dir={Option. Set path to parent directory of the label images. If you do not have the ground truth, no need to set this argument.} \
      --output_dir={Option. Set path to the output directory. If you do not set this argument, output files to ./output/seg_data/.}
  ```
- --segmentation_gt_json_dir

The directory consists of glomerular images and label data. See below.
```bash
+-- <--segmentation_gt_json_dir>
    +-- <Patient number>
        +-- glomerular_image_name.PNG: glomerular images
        +-- glomerular_image_name.json: label data
```

The json format is the same as the output [labelme](https://github.com/wkentaro/labelme).

- --object_detection_gt_xml_dir

The directory consists of glomerular detection ground truth files. See below.
```bash
+-- <--object_detection_gt_xml_dir>
    +-- 02_PAS/
        +-- <Patient number>
            +-- annotations
                +-- filename.xml: a ground truth file of glomerular detection
```

- --segmentation_gt_png_dir

The directory consists of glomerular label images (ground truth). See below.
```bash
+-- <segmentation_gt_png_dir>
    +-- all
        +-- <Patient number>
            +-- glomerular_label_name.PNG: glomerular label images
```
- --output_dir

This program outputs files below in "output_dir".
```bash
+-- <output_dir>
    +-- org_image
    |   +-- all
    |       +-- <Patient number>
    |           +-- glomerular_cropped_name.PNG: glomerular cropped images of detected areas
    +-- label (if you set segmentation_gt_json_dir, object_detection_gt_xml_dir, and segmentation_gt_png_dir.)
        +-- all
            +-- <Patient number>
                +-- glomerular_label_name.PNG: glomerular label images of detected areas
```
### 2. Glomerular segmentation on the detected region

```bash
python /opt/ESPNet/test/VisualizeResults_iou.py \
    --classes 5 \
    --rgb_data_dir {Requied. Set path to parent directory of original glomerular images} \
    --label_data_dir {Option. Set path to parent directory of label images if you want to evaluate accuracy} \
    --savedir {Option. Set path to the output directory of the results} \
    --weights {Required. Set path to the weights file. (e.g., /models/espnet_fold1.pth )} \
    --gpu_id {Option. Set gpu id. If -1, then CPU.} \
    --img_extn PNG \
    --mean {Required. Set gloabal mean values (BGR) of training images. See the table below.} \
    --std {Required. Set global standard deviation values (BGR) of training images. See the table below.} \
    --decoder \
    --colored \
    --overlay \
    --cityFormat
```
set mean and std RGB values of training dataset to the argument of "mean" and "std"  below when use our trained models. 

| fold | mean (B G R) | std (B G R) |
| --- | --- | --- |
| fold1 | 204.60071 170.19359 199.57469 | 20.61257 42.92207 28.401505 |
| fold2 | 202.38148 167.13171 198.10599 | 20.704079, 42.958416, 28.366297|
| fold3 | 203.12099 167.813 198.50894 | 21.038654 43.769535 29.034416 |
| fold4 | 203.66399 167.94217 198.58081 | 20.96783 43.556736 28.838718 |
| fold5 | 204.49896 169.03307 199.22058 | 20.547842 42.86628 27.966227 |

This program outputs files below in "savedir".

```bash
+-- <savedir>
    +-- <Patient number>
    |    +-- {input image name}_org.png: original images
    |    +-- {input image name}_overlay.png: predicted label overlayed on original images
    |    +-- {input image name}.json: predicted label images
    +-- combined_images
    |    +-- <Patient number>
    |        +-- {input image name}.png: combined images. Ordeir is <original image, ground-truth, predicted label overlayes on original image> 
    +-- summary_accuracy.csv: IoU of each category and mIoU are shown. This is generated if you set label_data_dir. Columns are "filename,glomerulus, crescent, sclerosis, mesangium, background iou,glomerulus iou,crescent iou,sclerosis iou, mesangium iou,mIoU"
    +-- summary_dataset.csv: summary of the ground truth. Number of glomeruli including each class is summarized. This is generated if you set label_data_dir. Columns are "patient_id, glomerulus, crescent, sclerosis, mesangium".
    +-- summary_pixel.csv: pixel summary of the predicted labels. Columns are "patient_id, filename, background, glomerulus, crescent, sclerosis, mesangium"
    +-- overall_accuracy.txt: output "overall_acc:{}, per_class_acc:{}, per_class_iou:{}, mIOU:{}". List is [background, glomerulus, crescent, sclerosis, mesangium].
```

### 3. Merge glomerular segmenation over WSIs and evaluate
```bash
python3 /opt/ESPNet/test/eval_wsi_segmentation.py \
    --staining=OPT_PAS
    --target_list={Required. Set path to the file of target list. In the file, ndpi filenames without extension are written. (txt file format)} \
    --merged_detection_result_csv={Required. Set path to the merged detection result (csv file format)} \
    --wsi_dir={Required. Set path to parent directory of whole slide images} \
    --segmentation_pred_json_dir={Required. Set path to the parent directory of the segmentation pred json files} \
    --segmentation_gt_json_dir={Option. Set path to parent directory of the segmentation grount truth json files. If you do not have the ground truth, no need to set this argument.} \
    --object_detection_gt_xml_dir={Option. Set path to the parent directory of the segmentation object detection xml filesi. If you do not have the ground truth, no need to set this argument.} \
    --segmentation_gt_png_dir={Option. Set path to parent directory of the label images. If you do not have the ground truth, no need to set this argument.} \
    --output_file={Option. Set output file name. If you do not set, output a file to ./output/seg_data_pred/seg_data_output.tsv.} \
    --output_dir={Option. Set path to the output directory for merged images. If you do not set, output a file to ./output/seg_data_pred/}
```
This program outputs files below in "output_dir".
```bash
+-- <output_dir>
    +-- {Patient number}_pred.jpg: overlayed predicted segmentation image
    +-- seg_data_output_tsv: IoU of each category and mIoU are shown. This is generated if you set --segmentation_gt_json_dir, --object_detection_gt_xml_dir, and --segmentation_gt_png_dir.
```
