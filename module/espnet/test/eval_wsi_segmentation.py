# -*- coding: utf-8 -*-
# Copyright 2020 The University of Tokyo Hospital. All Rights Reserved.
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This program is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

from annotation_handler import AnnotationHandler
import argparse
from collections import OrderedDict
import csv
import cv2
import glob
from IOUEval import iouEval
import json
from labelme import utils
import numpy as np
import os
import openslide
from PIL import Image
import re
import xml.etree.ElementTree as ElementTree
from utils import shape

MAGNIFICATION=8
PALLETE = [[0, 0, 0],
           [255, 0, 0],
           [0,184, 0],
           [255, 255, 0],
           [0,0,255],
           [128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],
           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],
           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],
           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]

def relabel(img):
    img[img == 13] = 4
    img[img == 12] = 3
    img[img == 11] = 2
    img[img == 8] = 1
    img[img == 7] = 0
    return img


class Generate_Segmentation_Gt(AnnotationHandler):
    """Evaluation and Visualization"""
    def __init__(self, staining_type, annotation_dir, target_list, detect_list_file,
                 iou_threshold, output_file, output_dir, wsi_dir, gt_png_dir, seg_gt_json_dir, window_size, seg_pred_json_dir, nclasses, no_save=False, start=0, end=0):
        super(Generate_Segmentation_Gt, self).__init__(annotation_dir, staining_type)
        self.MARGIN = 20 # 20 micrometre
        self.iou_threshold = iou_threshold
        self.detect_list_file = detect_list_file
        self.output_file = output_file
        self.output_dir = output_dir
        self.image_ext = ['.PNG', '.png']
        self.detected_glomus_list = {}
        self.detected_patient_id = []
        self.image = None
        self.overlap_d = {} # overlap_list  #key: date, value: [{"gt":gt, "pred":found_rect, "iou": iou, "json": json_file_name_l[0]}]
        self.seg_gt_json_dir = seg_gt_json_dir
        self.seg_pred_json_dir = seg_pred_json_dir
        self.wsi_dir = wsi_dir
        self.window_size = window_size
        self.annotation_file_date_pattern = '^\d{8}_(.+)'
        self.re_annotation_file_date_pattern = re.compile(self.annotation_file_date_pattern)
        self.glomus_category = ['glomerulus', 'glomerulus-kana']

        '''Flag indicating not saving visualization result.'''
        self.no_save = no_save

        self.target_list = target_list
        self.start = start
        self.end = end
        od = OrderedDict()
        od['glomerulus'] = 1
        od['crescent'] =  2
        od['collapsing'] = 3
        od['sclerosis'] = 3
        od['mesangium'] = 4
        od['poler_mesangium'] = 4
        self.target_dic = {'all': od}
        self.nclasses = nclasses
        self.iouEvalVal = iouEval(self.nclasses)

    def scan_files(self):
        """
            Abstract:
                Find the annotation file and do the following.
                1. Read GT(correct answer) of glomerular region.
                2. Read the glomeruli detection result file.
        """

        self.print_header()
        with open(self.target_list, "r") as list_file:
            lines = list_file.readlines()
            if self.end == 0 or self.end > len(lines):
                end = len(lines)
            else:
                end = self.end
            with open(self.output_file, "w") as out_f:
                for i in range(self.start, end):
                    [patient_id, file_body] = lines[i].split(os.sep)
                    annotation_dir = os.path.join(self.annotation_dir, self.staining_dir)
                    dir_path = os.path.join(annotation_dir, patient_id)
                    print("Analyzing :{}".format(patient_id))
                    if os.path.isdir(dir_path):
                        for file_name in os.listdir(os.path.join(dir_path, 'annotations')):
                            if os.path.isfile(os.path.join(os.path.join(dir_path, 'annotations'), file_name)):
                                body, ext = os.path.splitext(file_name)
                                if ext == '.xml' and file_name.find(self.staining_type) == 0:
                                    body_list = self.repattern.findall(body)
                                    slide_name_body = body_list[0][0].replace(self.staining_type + '_' + patient_id + '_', '')
                                    '''There are cases in which date information may be attached at the beginning of a slide name.
                                    If it is attached, delete it.'''
                                    slide_name_body_list = self.re_annotation_file_date_pattern.findall(slide_name_body)
                                    slide_name_body = slide_name_body[:9]
                                    if len(slide_name_body_list) == 1:
                                        slide_name_body = slide_name_body_list[0]
                                    if slide_name_body in self.detected_glomus_list:
                                        del self.gt_list[:]
                                        '''Read the image file corresponding to the annotation file.'''
                                        self.read_image(dir_path, body)
                                        try:
                                            self.read_annotation(os.path.join(dir_path, 'annotations'), file_name)
                                        except ElementTree.ParseError as e:
                                            print('{} is not well-formed:{}').format(file_name, e)
                                        # generate WSIs of prediction and ground truth, and evaluate the performance
                                        overall_acc, per_class_acc, per_class_iou, mIOU = self.generate_wsi_pred_gt_and_eval(slide_name_body, int(body_list[0][2]))
                                        out_f.write("{}\t{}\t{}\t{}\t{}\n".format(patient_id, overall_acc, per_class_acc, per_class_iou, mIOU))
                                        print("{}\t{}\t{}\t{}\t{}".format(patient_id, overall_acc, per_class_acc, per_class_iou, mIOU))
                overall_acc, per_class_acc, per_class_iou, mIOU = self.iouEvalVal.getMetricRight()
                out_f.write("total\t{}\t{}\t{}\t{}".format(overall_acc, per_class_acc, per_class_iou, mIOU))
                                   

    def print_result_record(self, body, recall, recall_hit_num, num_gt, num_detected):
        print('"{}",{},{},{},{}'.format(body, recall, recall_hit_num, num_gt, num_detected))

    def print_header(self):
        print('data,recall,recall_hit_num,gt_num,detect_num')

    def save_image(self, path, file_name):
        if not self.no_save:
            self.image.save(os.path.join(path, file_name))

    def generate_wsi_pred_gt_and_eval(self, file_key, times):
        """
            Abstract: generate WSIs of prediction and ground truth, and evaluate the performance
            Args:
                file_key: [str] directionary name ex) H16-09557
                times: [int] 
        """
        seg_gt_json_l = glob.glob(os.path.join(self.seg_gt_json_dir, file_key, "*.json"))
        seg_pred_json_l = glob.glob(os.path.join(self.seg_pred_json_dir, file_key, "*.json"))
        ndpi_l = glob.glob(os.path.join(self.wsi_dir, file_key, "*ndpi"))
        assert len(ndpi_l) == 1
        ndpi_path_s = ndpi_l[0]
        margin_x, margin_y, slide_width, slide_height = self.read_slide_and_cal_margin(ndpi_path_s)
        iou_eval_val = iouEval(self.nclasses)
        # make numpy for prediction and ground truth of WSI
        whole_gt_np = np.zeros((int(slide_height/MAGNIFICATION), int(slide_width/MAGNIFICATION), 3), dtype=int)
        whole_pred_np = np.zeros((int(slide_height/MAGNIFICATION), int(slide_width/MAGNIFICATION), 3), dtype=int)
        # make ground truth and prediction of window, and confusion_matrix
        for x_ind in range(slide_width//self.window_size + 1):
            xmin = x_ind * self.window_size
            if x_ind == slide_width//self.window_size:
                xmax = slide_width
            else:
                xmax = (x_ind + 1) * self.window_size
            if xmax > slide_width:
                continue
            for y_ind in range(slide_height//self.window_size + 1):
                ymin = y_ind * self.window_size
                if y_ind == slide_height//self.window_size:
                    ymax = slide_height
                else:
                    ymax = (y_ind + 1) * self.window_size
                if ymax > slide_width:
                    continue
                # make ground truth
                gt_np = self.overlay(self.gt_list, times, margin_x, margin_y, seg_gt_json_l, xmin, ymin, xmax, ymax, "gt")
                # make prediction result
                pred_np = self.overlay(self.detected_glomus_list[file_key], 1, 0, 0, seg_pred_json_l, xmin, ymin, xmax, ymax, "pred")
                # make confusion matrix of ground truth and prediction result 
                iou_eval_val.addBatch(pred_np, gt_np)
                self.iouEvalVal.addBatch(pred_np, gt_np)
                # make ground truth and prediction result of WSI
                whole_gt_np = self.generate_whole_img([xmin, ymin, xmax, ymax], whole_gt_np, gt_np)
                whole_pred_np = self.generate_whole_img([xmin, ymin, xmax, ymax], whole_pred_np, pred_np)
        # save images
        output_gt_file_name = os.path.join(self.output_dir, file_key + "_gt.jpg")
        output_pred_file_name = os.path.join(self.output_dir, file_key + "_pred.jpg")
        cv2.imwrite(output_gt_file_name, whole_gt_np)
        cv2.imwrite(output_pred_file_name, whole_pred_np)
        # calcurate performance
        overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval_val.getMetricRight()
        return overall_acc, per_class_acc, per_class_iou, mIOU

    def generate_whole_img(self, bbox_l, whole_img_np, label_img_np):
        """
            Abstract: generate whole slide image
            Argument:
                bbox_l: <list> coordinate in order of xmin, ymin, xmax, ymax
                whole_img_np: <numpy> a whole original image
                label_img_np: <numpy> a whole label image
            Return:
                whole_img_np: <numpy> a whole slide image overlaid with a label image
        """
        width = bbox_l[2] - bbox_l[0]
        height = bbox_l[3] - bbox_l[1]
        target_region_pil = self.slide.read_region((bbox_l[0], bbox_l[1]), 0, (bbox_l[2]-bbox_l[0], bbox_l[3]-bbox_l[1]))
        target_region_np = np.asarray(target_region_pil.convert("RGB"))
        target_region_np = cv2.resize(target_region_np, (int(width/MAGNIFICATION), int(height/MAGNIFICATION)), interpolation=cv2.INTER_NEAREST)
        label_img_np = cv2.resize(label_img_np,  (int(width/MAGNIFICATION), int(height/MAGNIFICATION)), interpolation=cv2.INTER_NEAREST)
        classMap_numpy_color = np.zeros((target_region_np.shape[0], target_region_np.shape[1], target_region_np.shape[2]), dtype=np.uint8)
        for idx in range(len(PALLETE)):
            [r, g, b] = PALLETE[idx]
            classMap_numpy_color[label_img_np == idx] = [b, g, r]
        overlayed = cv2.addWeighted(target_region_np, 0.4, classMap_numpy_color, 0.6, 0)
        xmin = int(bbox_l[0]//MAGNIFICATION)
        ymin = int(bbox_l[1]//MAGNIFICATION)
        xmax = int(bbox_l[2]//MAGNIFICATION)
        ymax = int(bbox_l[3]//MAGNIFICATION)
        whole_img_np[ymin:ymax, xmin:xmax,:] = overlayed
        return whole_img_np
 
    def overlay(self, bbox_list, times, margin_x, margin_y, seg_bbox_json_l, xmin, ymin, xmax, ymax, data_type):
        """
            Abstract: overlay predicted segmentation on a window area
            Argument: 
                bbox_list: list of bounding box coordinate of ground truth or prediction
                margin_x: margin for ground truth of segmentation in x direction
                margin_y: margin for ground truth of segmentation in y direction
                seg_bbox_json_l: <list> 
                xmin: <int> minimum x coordinate of window
                ymin: <int> minimum y coordinate of window
                xmax: <int> max x coordinate of window
                ymax: <int> max x coordinate of window
                data_type: <str> gt, other. 
            Return:
                window_np: <numpy.ndarray> overlaid window area
        """
        window_np = np.zeros((ymax-ymin, xmax-xmin), dtype=int)
        for gt in bbox_list:
            gt_l = list(map(lambda x: x * times, gt))
            gt_margin_l = []
            gt_margin_l.append(int(gt_l[0] - margin_x))
            gt_margin_l.append(int(gt_l[1] - margin_y))
            gt_margin_l.append(int(gt_l[2] + 2 * margin_x)) # extra right side due to bug of clip_annotated_area.py
            gt_margin_l.append(int(gt_l[3] + 2 * margin_y)) # extra right side due to bug of clip_annotated_area.py
            gt_margin_l = list(map(lambda x: x, gt_margin_l))
            iou = self.check_overlap([xmin, ymin, xmax, ymax], gt_l)
            if iou > 0.0:
                # overlay predicted segmentation on the window area
                # search json file
                search_name = "xmin{}_ymin{}_xmax{}_ymax{}".format(int(gt_l[0]/8), int(gt_l[1]/8), int(gt_l[2]/8), int(gt_l[3]/8))
                json_file_name_l = [json_name for json_name in seg_bbox_json_l if re.search(search_name, json_name)]
                assert len(json_file_name_l) <= 1 
                if len(json_file_name_l) == 0:
                    # 0: there is no ground truth json files. This is because the glomerulous is blurred, so, the glomerular image is not annotated. Or, you forgot implementing make_data.py
                    continue
                data = json.load(open(json_file_name_l[0]), object_pairs_hook=OrderedDict)
                if data['imageData']:
                    imageData = data['imageData']
                else:
                    imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
                    with open(imagePath, 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')

                img = utils.img_b64_to_arr(imageData)

                label_name_to_value = {'_background_': 0}
                target_list = self.target_dic["all"]
                for target in target_list:
                    label_name_to_value[target] = target_list[target]
                label_name_list = target_list
                if data_type=="gt":
                    img = shape.shapes_to_label(img.shape, data['shapes'], label_name_to_value, label_name_list)
                else:
                    img = relabel(img)
                # extract coordinate of the union of window and ground truth
                whole_area_xmin = min([xmin, gt_margin_l[0]])
                whole_area_ymin = min([ymin, gt_margin_l[1]])
                whole_area_xmax = max([xmax, gt_margin_l[2]])
                whole_area_ymax = max([ymax, gt_margin_l[3]])
                whole_area_np = np.zeros((int(whole_area_ymax - whole_area_ymin), int(whole_area_xmax - whole_area_xmin)), dtype=int)
                # overlay ground truth on the union of window and margin_gt
                whole_area_np[gt_margin_l[1]-whole_area_ymin:gt_margin_l[3]-whole_area_ymin, gt_margin_l[0]-whole_area_xmin:gt_margin_l[2]-whole_area_xmin] = img
                # extract window area
                print("xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(xmin, ymin, xmax, ymax))
                print("whole_area_xmin:{}, whole_area_min:{}, whole_area_xmax:{}, whole_area_ymax:{}".format(whole_area_xmin, whole_area_ymin, whole_area_xmax, whole_area_ymax))
                window_np = np.asarray((window_np, whole_area_np[ymin-whole_area_ymin:ymax-whole_area_ymin, xmin-whole_area_xmin:xmax-whole_area_xmin]), dtype=int)
                window_np = np.max(window_np, axis=0)
                assert window_np.shape[0] == ymax - ymin
                assert window_np.shape[1] == xmax - xmin
                assert np.max(window_np.flatten()) < self.nclasses
        return window_np

    def read_detected_glomus_list(self):
        """Read patient number and coordinate of the detected area"""
        with open(self.detect_list_file, "r") as list_file:
            file_body = ''
            reader = csv.reader(list_file)
            for row in reader:
                body = row[1].replace(' ', '')
                if file_body != body:
                    file_body = body
                    self.detected_glomus_list[file_body] = []
                    self.detected_patient_id.append(row[1])

                self.detected_glomus_list[file_body].append([int(row[3]), int(row[4]), int(row[5]), int(row[6]), float(row[7])])

    def read_image(self, dir_path, body):
        """Read target image"""
        for ext in self.image_ext:
            file_path = os.path.join(dir_path, body + ext)
            if os.path.isfile(file_path):
                self.image = Image.open(file_path)
                break
                
    def read_slide_and_cal_margin(self, ndpi_file_path):
        """
            Abstract: calculate margin for segmentation
            Argument: 
                ndpi_file_path: <str> path to ndpi file
            Return:
                margin_x: <int> margin for ground truth of segmentation in x direction
                margin_y: <int> margin for ground truth of segmentation in y direction
                slide_width: <int> x size of the wsi
                slide_height: <int> y size of the wsi
        """
        self.slide = openslide.open_slide(ndpi_file_path)
        slide_width, slide_height = self.slide.dimensions
        mpp_x = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        margin_x = int(round(float(self.MARGIN) / mpp_x))
        margin_y = int(round(float(self.MARGIN) / mpp_y))
        return margin_x, margin_y, slide_width, slide_height

    def generate_pred_wsi(self):
        """
            Abstract: generate prediction wsi
        """
        for file_key in self.detected_glomus_list.keys(): #file_key: [str] directionary name ex) H16-09557
            seg_pred_json_l = glob.glob(os.path.join(self.seg_pred_json_dir, file_key, "*.json"))
            ndpi_l = glob.glob(os.path.join(self.wsi_dir, file_key, "*ndpi"))
            assert len(ndpi_l) == 1
            ndpi_path_s = ndpi_l[0]
            margin_x, margin_y, slide_width, slide_height = self.read_slide_and_cal_margin(ndpi_path_s)
            # make numpy for prediction and ground truth of WSI
            whole_pred_np = np.zeros((int(slide_height/MAGNIFICATION), int(slide_width/MAGNIFICATION), 3), dtype=int)
            # make ground truth and prediction of window, and confusion_matrix
            for x_ind in range(slide_width//self.window_size + 1):
                xmin = x_ind * self.window_size
                if x_ind == slide_width//self.window_size:
                    xmax = slide_width
                else:
                    xmax = (x_ind + 1) * self.window_size
                if xmax > slide_width:
                    continue
                for y_ind in range(slide_height//self.window_size + 1):
                    ymin = y_ind * self.window_size
                    if y_ind == slide_height//self.window_size:
                        ymax = slide_height
                    else:
                        ymax = (y_ind + 1) * self.window_size
                    if ymax > slide_width:
                        continue
                    # make prediction result
                    pred_np = self.overlay(self.detected_glomus_list[file_key], 1, 0, 0, seg_pred_json_l, xmin, ymin, xmax, ymax, "pred")
                    # make ground truth and prediction result of WSI
                    whole_pred_np = self.generate_whole_img([xmin, ymin, xmax, ymax], whole_pred_np, pred_np)
            # save images
            output_pred_file_name = os.path.join(self.output_dir, file_key + "_pred.jpg")
            cv2.imwrite(output_pred_file_name, whole_pred_np)


def parse_args():
    '''
        Abstract: 
            Parse input arguments
        return: 
            args
    '''
    parser = argparse.ArgumentParser(description='merge cropped glomerular segmented images')
    parser.add_argument('--staining', dest='staining', help="Set staining method. e.g. OPT_PAS", type=str, required=True)
    parser.add_argument('--merged_detection_result_csv', dest='input_csv', help="Set path to the merged detection result (csv file format)", type=str, required=True)
    parser.add_argument('--target_list', dest='target_list', help="Set path to the file of target list. In the file, ndpi filenames without extension are written. (txt file format)", type=str, required=True)
    parser.add_argument('--wsi_dir', dest='wsi_dir', help="Set path to parent directory of whole slide images", type=str, required=True)
    parser.add_argument('--segmentation_pred_json_dir', dest='seg_pred_json_dir', help="Set path to the parent directory of the segmentation pred json files", type=str, required=True)
    parser.add_argument('--object_detection_gt_xml_dir', dest='ob_gt_xml_dir', help="Set path to the parent directory of the segmentation object detection xml filesi. If you do not have the ground truth, no need to set this argument.", type=str, default=None)
    parser.add_argument('--segmentation_gt_json_dir', dest='seg_gt_json_dir', help="Set path to parent directory of the segmentation grount truth json files. If you do not have the ground truth, no need to set this argument.", type=str, default=None)
    parser.add_argument('--iou_threshold', dest='iou_threshold', help="Set iou threshold", type=float, default=0.01)
    parser.add_argument('--output_file', dest='output_file', help="Set output file name. If you do not set, output a file to ./output/seg_data_pred/seg_data_output.tsv.", type=str, default='./output/seg_data_pred/seg_data_output.tsv')
    parser.add_argument('--output_dir', dest='output_dir', help="Set path to the output directory for merged images. If you do not set, output a file to ./output/seg_data_pred/", type=str, default='./output/seg_data_pred')
    parser.add_argument('--start', dest='start', help="Set --start for start line begin 0", type=int, default=0)
    parser.add_argument('--end', dest='end', help="Set --end for end line(含まれない）", type=int, default=0)
    parser.add_argument('--window_size', dest='window_size', help="Set window size", type=int, default=2400)
    parser.add_argument('--segmentation_gt_png_dir', dest='gt_png_dir', help="Set path to parent directory of the label images. If you do not have the ground truth, no need to set this argument.", type=str, default=None)
    parser.add_argument('--no_save', dest='no_save', help="Set --no_save for test", action='store_true')
    parser.add_argument('--classes', dest='classes', help="Set number of classes", type=int, default=5)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = Generate_Segmentation_Gt(args.staining, args.ob_gt_xml_dir, args.target_list, args.input_csv, args.iou_threshold, args.output_file, args.output_dir, args.wsi_dir, args.gt_png_dir, args.seg_gt_json_dir, args.window_size, args.seg_pred_json_dir, args.classes, args.no_save, args.start, args.end)
    model.read_detected_glomus_list()
    if args.seg_gt_json_dir is None or args.gt_png_dir is None or args.ob_gt_xml_dir is None:
        # generate prediction wsi
        model.generate_pred_wsi()
    else:
        # generate prediction and gt wsi, and evaluate
        model.scan_files()
