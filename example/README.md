# Quick Start
Examples of glomerular segmentation on WSI.
Follow the guide below.
## Run docker container
```bash
# If you use NVIDIA Docker
nvidia-docker run --rm -ti -v <path to your directory>:/workspace glomerular-espnet-pytorch:latest /bin/bash
```
```bash
# If you use NVIDIA GPUs
docker run --runtime=nvidia --rm -ti -v <path to your directory>:/workspace glomerular-espnet-pytorch:latest /bin/bash
```
## Export environment variable as you would like
```bash
# define output directory for object detection and segmentation
output_dir=/workspace/output
```
## Glomerular detection
### Download trained models
```bash
cd /workspace;
wget https://ai-health.m.u-tokyo.ac.jp/labweb/dl/renal_ai/segmentation/finetuned_frcnn.tar.gz;
tar -zxvf finetuned_frcnn.tar.gz
```
### Glomerular detection
```bash
python /opt/glomeruli_detection/detect_glomus_test.py \
    --model=/workspace/fold_1 \
    --target_list=/opt/ESPNet/example/opt_pas_test_list.txt \
    --data_dir=/opt/ESPNet/example/data \
    --staining=OPT_PAS \
    --output_dir=${output_dir} \
    --output_file_ext=_test1 \
    --window_size=2000 \
    --overlap_ratio=0.1 \
    --conf_threshold=0.2 \
    --model_name=frozen_inference_graph.pb
```
### Merge glomeruli bounding boexes
```bash
python /opt/glomeruli_detection/merge_overlaped_glomus.py \
    --target_list=/opt/ESPNet/example/opt_pas_test_list.txt \
    --detected_list=/workspace/output/OPT_PAS_test1.csv \
    --data_dir=/opt/ESPNet/example/data \
    --staining=OPT_PAS \
    --output_dir=/workspace/output \
    --output_file_ext=test1 \
    --conf_threshold=0.9 \
    --overlap_threshold=0.35
```
## Glomerular segmentation
### Generate cropped images based on the prediction of faster-rcnn
```bash
# If you have ground truth of segmentation
python /opt/glomeruli_detection/make_seg_data.py \
    --staining=OPT_PAS \
    --target_list=/opt/ESPNet/example/opt_pas_test_list.txt \
    --merged_detection_result_csv=${output_dir}/OPT_PAS_GlomusMergedList_test1.csv \
    --segmentation_gt_json_dir=/opt/ESPNet/example/data/seg_annotation \
    --object_detection_gt_xml_dir=/opt/ESPNet/example/data \
    --wsi_dir=/opt/ESPNet/example/data/02_PAS \
    --segmentation_gt_png_dir=/opt/ESPNet/example/data/label \
    --output_dir=${output_dir}/seg_data
```
```bash
# If you do not have ground truth of segmentation
python /opt/glomeruli_detection/make_seg_data.py \
    --staining=OPT_PAS \
    --target_list=/opt/ESPNet/example/opt_pas_test_list.txt \
    --merged_detection_result_csv=${output_dir}/OPT_PAS_GlomusMergedList_test1.csv \
    --wsi_dir=/opt/ESPNet/example/data/02_PAS \
    --output_dir=${output_dir}/seg_data
```
### Segment detected area
```bash
# If you have ground truth of segmentation
python /opt/ESPNet/test/VisualizeResults_iou_pixel.py \
    --classes=5 \
    --rgb_data_dir=${output_dir}/seg_data/org_image \
    --label_data_dir=${output_dir}/seg_data/label/all \
    --savedir=${output_dir}/seg_data_pred \
    --weights=/opt/ESPNet/models/espnet_fold1.pth \
    --gpu_id=0 \
    --decoder \
    --img_extn=PNG \
    --colored \
    --overlay \
    --mean 204.60071 170.19359 199.57469 \
    --std 20.61257 42.92207 28.401505
```
```bash
# If you do not have ground truth of segmentation
python /opt/ESPNet/test/VisualizeResults_iou.py \
    --classes=5 \
    --rgb_data_dir=${output_dir}/seg_data/org_image \
    --savedir=${output_dir}/seg_data_pred \
    --weights=/opt/ESPNet/models/espnet_fold1.pth \
    --gpu_id=0 \
    --decoder \
    --img_extn=PNG \
    --colored \
    --overlay \
    --mean 204.60071 170.19359 199.57469 \
    --std 20.61257 42.92207 28.401505
```
### Merge cropped images and evaluate it
```bash
# If you have ground truth of segmentation
python /opt/ESPNet/test/eval_wsi_segmentation.py \
    --staining=OPT_PAS \
    --target_list=/opt/ESPNet/example/opt_pas_test_list.txt \
    --merged_detection_result_csv=${output_dir}/OPT_PAS_GlomusMergedList_test1.csv \
    --segmentation_gt_json_dir=/opt/ESPNet/example/data/seg_annotation \
    --object_detection_gt_xml_dir=/opt/ESPNet/example/data \
    --wsi_dir=/opt/ESPNet/example/data/02_PAS \
    --segmentation_gt_png_dir=/opt/ESPNet/example/data/label \
    --output_file=${output_dir}/seg_data_pred/seg_data_output.tsv \
    --segmentation_pred_json_dir=${output_dir}/seg_data_pred \
    --window_size=2400 \
    --output_dir=${output_dir}/seg_data_pred
```
```bash
# If you do not have ground truth of segmentation
python /opt/ESPNet/test/eval_wsi_segmentation.py \
    --staining=OPT_PAS \
    --target_list=/opt/ESPNet/example/opt_pas_test_list.txt \
    --merged_detection_result_csv=${output_dir}/OPT_PAS_GlomusMergedList_test1.csv \
    --wsi_dir=/opt/ESPNet/example/data/02_PAS \
    --output_file=${output_dir}/seg_data_pred/seg_data_output.tsv \
    --segmentation_pred_json_dir=${output_dir}/seg_data_pred \
    --window_size=2400 \
    --output_dir=${output_dir}/seg_data_pred
```
