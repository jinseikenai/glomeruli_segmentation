# -*- coding: utf-8 -*-
# Copyright 2020 The University of Tokyo Hospital. All Rights Reserved.
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This program is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

from module.common.annotation_handler import AnnotationHandler
import argparse
from collections import OrderedDict
import csv
import cv2
import glob
from module.common.IOUEval import iouEval
import json
# from labelme import utils
import numpy as np
import os
import openslide
from PIL import Image
import re
import xml.etree.ElementTree as ElementTree
# from module.common.utils import shape
# from module.common.color_palette import palette

MAGNIFICATION = 8
# PALETTE = [[0, 0, 0], [120, 120, 120], [213, 47, 0],  [0, 180, 58], [43, 90, 233], [255, 255, 128]]
# overlay させるために色を少し強調しておく
PALETTE = [[0, 0, 0], [120, 120, 120], [250, 47, 0],  [0, 220, 58], [43, 90, 250], [255, 255, 100]]


class Generate_Segmentation_Gt(AnnotationHandler):
    """Evaluation and Visualization"""
    def __init__(self, staining_type, annotation_dir, target_list, detect_list_file,
                 iou_threshold, output_file, output_dir, wsi_dir, seg_gt_image_dir, window_size,
                 seg_pred_image_dir, nclasses, no_save=False, start=0, end=0):
        super(Generate_Segmentation_Gt, self).__init__(annotation_dir, staining_type)
        self.MARGIN = 20  # 20 micrometre
        self.iou_threshold = iou_threshold
        self.detect_list_file = detect_list_file
        self.output_file = output_file
        self.output_dir = output_dir
        self.image_ext = '.PNG'
        self.detected_glomus_list = {}
        self.detected_patient_id = []
        self.image = None
        self.overlap_d = {} # overlap_list  #key: date, value: [{"gt":gt, "pred":found_rect, "iou": iou, "json": json_file_name_l[0]}]
        self.seg_gt_image_dir = seg_gt_image_dir
        self.seg_pred_image_dir = seg_pred_image_dir
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
        od['tuft'] = 2
        od['crescent'] = 3
        od['collapsing'] = 4
        od['sclerosis'] = 4
        # od['mesangium'] = 4
        # od['poler_mesangium'] = 4
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
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)
            with open(self.output_file, "w") as out_f:
                for i in range(self.start, end):
                    patient_id, file_body = lines[i].strip().split(',')[0].split(os.sep)
                    # slide_name_body = body_list[0][0].replace(self.staining_type + '_' + patient_id + '_', '')
                    '''There are cases in which date information may be attached at the beginning of a slide name.
                    If it is attached, delete it.'''
                    # slide_name_body_list = self.re_annotation_file_date_pattern.findall(slide_name_body)
                    # slide_name_body = slide_name_body[:9]
                    slide_name_body = patient_id
                    # if len(slide_name_body_list) == 1:
                    #     slide_name_body = slide_name_body_list[0]
                    if slide_name_body in self.detected_glomus_list:
                        print("Analyzing :{}".format(patient_id))
                        del self.gt_list[:]
                        '''Read the image file corresponding to the annotation file.'''
                        # self.read_image(os.path.join(args.seg_gt_image_dir, slide_name_body), file_body)
                        # try:
                        #     self.read_annotation(os.path.join(dir_path, 'annotations'), file_name)
                        # except ElementTree.ParseError as e:
                        #     print('{} is not well-formed:{}').format(file_name, e)
                        # generate WSIs of prediction and ground truth, and evaluate the performance
                        overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, mDice = self.generate_wsi_pred_gt_and_eval(slide_name_body)
                        out_f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(patient_id, overall_acc, per_class_acc,
                                                                          per_class_iou, mIOU,
                                                                          per_class_dice, mDice))
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(patient_id, overall_acc, per_class_acc,
                                                                  per_class_iou, mIOU,
                                                                  per_class_dice, mDice))
                overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, mDice = self.iouEvalVal.getMetricMicro()
                out_f.write("total\t{}\t{}\t{}\t{}\t{}\t{}".format(overall_acc, per_class_acc, per_class_iou, mIOU,
                                                                   per_class_dice, mDice))
                                   

    def print_result_record(self, body, recall, recall_hit_num, num_gt, num_detected):
        print('"{}",{},{},{},{}'.format(body, recall, recall_hit_num, num_gt, num_detected))

    def print_header(self):
        print('data,recall,recall_hit_num,gt_num,detect_num')

    def save_image(self, path, file_name):
        if not self.no_save:
            self.image.save(os.path.join(path, file_name))

    def generate_wsi_pred_gt_and_eval(self, file_key):
        """
            Abstract: generate WSIs of prediction and ground truth, and evaluate the performance
            Args:
                file_key: [str] directionary name ex) H16-09557
                times: [int] 
        """
        # seg_gt_json_l = glob.glob(os.path.join(self.seg_gt_json_dir, file_key, "*.json"))
        seg_gt_l = glob.glob(os.path.join(self.seg_gt_image_dir, file_key, "*.PNG"))
        self.gt_list = []
        self.read_gt_list(seg_gt_l)
        seg_pred_l = glob.glob(os.path.join(self.seg_pred_image_dir, file_key, "*.PNG"))
        ndpi_l = glob.glob(os.path.join(self.wsi_dir, file_key, "*ndpi"))
        assert len(ndpi_l) == 1
        ndpi_path_s = ndpi_l[0]
        margin_x, margin_y, slide_width, slide_height = self.read_slide_and_cal_margin(ndpi_path_s)
        iou_eval_val = iouEval(self.nclasses)
        # make numpy for prediction and ground truth of WSI
        whole_gt_np = np.zeros((int(slide_height/MAGNIFICATION), int(slide_width/MAGNIFICATION), 3), dtype=int)
        whole_pred_np = np.zeros((int(slide_height/MAGNIFICATION), int(slide_width/MAGNIFICATION), 3), dtype=int)
        # make ground truth and prediction images of WSI(each by splitted window), and confusion_matrix
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

                if int((xmax - xmin)/MAGNIFICATION) <= 0 or int((ymax - ymin)/MAGNIFICATION) <= 0:
                    continue
                print('{} block x:{}/{}, y:{}/{}'.format(file_key, x_ind + 1, slide_width//self.window_size + 1,
                                                         y_ind + 1, slide_height//self.window_size + 1))
                # make ground truth
                gt_np = self.overlay(self.gt_list, 1, margin_x, margin_y, seg_gt_l, xmin, ymin, xmax, ymax, "gt")
                # make prediction result
                pred_np = self.overlay(self.detected_glomus_list[file_key], 1, margin_x, margin_y, seg_pred_l, xmin, ymin, xmax, ymax, "pred")
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
        overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, mDice = iou_eval_val.getMetricMicro()
        return overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, mDice

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
        for idx in range(len(PALETTE)):
            [r, g, b] = PALETTE[idx]
            classMap_numpy_color[label_img_np == idx] = [b, g, r]
        # overlayed = cv2.addWeighted(target_region_np, 0.5, classMap_numpy_color, 0.5, 0)
        overlayed = cv2.addWeighted(target_region_np, 0.4, classMap_numpy_color, 0.6, 0)
        xmin = int(bbox_l[0]//MAGNIFICATION)
        ymin = int(bbox_l[1]//MAGNIFICATION)
        xmax = int(bbox_l[2]//MAGNIFICATION)
        ymax = int(bbox_l[3]//MAGNIFICATION)
        whole_img_np[ymin:ymax, xmin:xmax, :] = overlayed
        return whole_img_np
 
    def overlay(self, bbox_list, times, margin_x, margin_y, seg_img_l, xmin, ymin, xmax, ymax, data_type):
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
        for seg in bbox_list:
            # gt_l = list(map(lambda x: x * times, gt))
            # seg[:4] = [x / times for x in seg[:4]]
            tmp_seg = [int(round(seg[0] / times)), int(round(seg[1] / times)), int(round(seg[2] / times)), int(round(seg[3] / times))]
            iou = self.check_overlap([xmin, ymin, xmax, ymax], seg)
            if iou > 0.0:
                # overlay predicted segmentation on the window area
                # search json file
                # search_name = "xmin{}_ymin{}_xmax{}_ymax{}".format(int(gt_l[0]/8), int(gt_l[1]/8), int(gt_l[2]/8), int(gt_l[3]/8))
                search_name = "xmin{}_ymin{}_xmax{}_ymax{}".format(int(tmp_seg[0]), int(tmp_seg[1]), int(tmp_seg[2]), int(tmp_seg[3]))
                # json_file_name_l = [json_name for json_name in seg_bbox_json_l if re.search(search_name, json_name)]
                matched_seg_image_l = [seg_img_name for seg_img_name in seg_img_l if re.search(search_name, seg_img_name)]
                assert len(matched_seg_image_l) <= 1
                if len(matched_seg_image_l) == 0:
                    # 0: there is no ground truth json files. This is because the glomerulous is blurred, so, the glomerular image is not annotated. Or, you forgot implementing make_data.py
                    continue
                '''
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
                # if data_type == "gt":
                img = shape.shapes_to_label(img.shape, data['shapes'], label_name_to_value, label_name_list)
                '''
                # else:
                # img = relabel(img)
                # if self.nclasses == 4:
                #     img = relabel_4cls(img)
                # extract coordinate of the union of window and ground truth
                seg_margin_l = [int(seg[0] - margin_x),
                                int(seg[1] - margin_y),
                                int(seg[2] + margin_x),
                                int(seg[3] + margin_y)]
                # seg_margin_l = list(map(lambda x: x, seg_margin_l))
                overlap_area = [max([xmin, seg_margin_l[0]]),
                                max([ymin, seg_margin_l[1]]),
                                min([xmax, seg_margin_l[2]]),
                                min([ymax, seg_margin_l[3]])]
                r_overlap_area = [overlap_area[0] - xmin, overlap_area[1] - ymin,
                                  overlap_area[2] - xmin, overlap_area[3] - ymin]
                # overlap_area_np = np.zeros((int(overlap_area[3] - overlap_area[1]),
                #                             int(overlap_area[2] - overlap_area[0])), dtype=int)
                # overlay ground truth on the union of window and margin_gt
                seg_img = Image.open(matched_seg_image_l[0])
                seg_img = np.asarray(seg_img, dtype=int)
                if seg_img.shape[0] != overlap_area[3] - overlap_area[1] or seg_img.shape[1] != overlap_area[2] - overlap_area[0]:
                    seg_img = seg_img[overlap_area[1] - seg_margin_l[1]: overlap_area[3] - seg_margin_l[1], overlap_area[0] - seg_margin_l[0]: overlap_area[2] - seg_margin_l[0]]
                # overlap_area_np[gt_margin_l[1]-whole_area_ymin:gt_margin_l[3]-whole_area_ymin, gt_margin_l[0]-whole_area_xmin:gt_margin_l[2]-whole_area_xmin] = seg
                # extract window area
                # print("xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(xmin, ymin, xmax, ymax))
                # print("whole_area_xmin:{}, whole_area_min:{}, whole_area_xmax:{}, whole_area_ymax:{}".format(whole_area_xmin, whole_area_ymin, whole_area_xmax, whole_area_ymax))
                print("find: {}:{}".format(data_type, search_name))
                # window_np = np.asarray((window_np, whole_area_np[ymin-whole_area_ymin:ymax-whole_area_ymin, xmin-whole_area_xmin:xmax-whole_area_xmin]), dtype=int)
                # window_np = np.max(window_np, axis=0)
                window_np[r_overlap_area[1]: r_overlap_area[3], r_overlap_area[0]: r_overlap_area[2]] = np.maximum(window_np[r_overlap_area[1]: r_overlap_area[3], r_overlap_area[0]: r_overlap_area[2]], seg_img)
                assert window_np.shape[0] == ymax - ymin
                assert window_np.shape[1] == xmax - xmin
                assert np.max(window_np.flatten()) < self.nclasses
        return window_np

    def read_detected_glomus_list(self):
        """Read patient number and coordinate of the detected area"""
        detected_file_list = glob.glob(os.path.join(self.seg_pred_image_dir, '*/*.PNG'))
        specimen_id_list = []
        for dt_file in detected_file_list:
            s_id = dt_file.split(os.path.sep)[-2]
            if s_id not in specimen_id_list:
                specimen_id_list.append(s_id)
        with open(self.detect_list_file, "r") as list_file:
            file_body = ''
            reader = csv.reader(list_file)
            for row in reader:
                body = row[1].replace(' ', '')
                if body in specimen_id_list:
                    if file_body != body:
                        file_body = body
                        self.detected_glomus_list[file_body] = []
                        self.detected_patient_id.append(row[1])

                    self.detected_glomus_list[file_body].append([int(row[3]), int(row[4]), int(row[5]), int(row[6]), float(row[7])])

    def read_gt_list(self, detected_file_list, times=1):
        for file_name in detected_file_list:
            file_name = os.path.basename(file_name)
            file_parts = os.path.splitext(file_name)[0].split('_')
            self.gt_list.append([int(file_parts[-4].lstrip('xmin'))*times, int(file_parts[-3].lstrip('ymin'))*times,
                                 int(file_parts[-2].lstrip('xmax'))*times, int(file_parts[-1].lstrip('ymax'))*times,
                                 1.0])

    def read_image(self, dir_path, body):
        """Read target image"""
        # for ext in self.image_ext:
        file_path = os.path.join(dir_path, body)
        if os.path.isfile(file_path):
            self.image = Image.open(file_path)
                
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
        with open(os.path.join(self.output_dir, self.output_file), "w") as out_f:
            for file_key in self.detected_glomus_list.keys(): #file_key: [str] directionary name ex) H16-09557
                # seg_pred_json_l = glob.glob(os.path.join(self.seg_pred_json_dir, file_key, "json/*.json"))
                seg_pred_l = glob.glob(os.path.join(self.seg_pred_image_dir, file_key, "*.PNG"))
                ndpi_l = glob.glob(os.path.join(self.wsi_dir, file_key, "*ndpi"))
                assert len(ndpi_l) == 1
                ndpi_path_s = ndpi_l[0]
                margin_x, margin_y, slide_width, slide_height = self.read_slide_and_cal_margin(ndpi_path_s)
                # make numpy for prediction and ground truth of WSI
                whole_gt_np = np.zeros((int(slide_height/MAGNIFICATION), int(slide_width/MAGNIFICATION), 3), dtype=int)
                whole_pred_np = np.zeros((int(slide_height/MAGNIFICATION), int(slide_width/MAGNIFICATION), 3), dtype=int)
                # make ground truth and prediction of window, and confusion_matrix
                iou_eval_val = iouEval(self.nclasses)
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
                        if int((xmax - xmin)/MAGNIFICATION) <= 0 or int((ymax - ymin)/MAGNIFICATION) <= 0:
                            continue
                        print('{} block x:{}/{}, y:{}/{}'.format(file_key, x_ind + 1, slide_width//self.window_size + 1,
                                                                 y_ind + 1, slide_height//self.window_size + 1))
                        # make ground truth
                        seg_gt_l = glob.glob(os.path.join(self.seg_gt_image_dir, file_key, "*.PNG"))
                        self.gt_list = []
                        # GTファイルの座標は1/8スケールで記録されている
                        self.read_gt_list(seg_gt_l, times=8)
                        gt_np = self.overlay(self.gt_list, 8, margin_x, margin_y, seg_gt_l, xmin, ymin, xmax, ymax, "gt")
                        # make prediction result
                        # pred_np = self.overlay(self.detected_glomus_list[file_key], 1, 0, 0, seg_pred_json_l, xmin, ymin, xmax, ymax, "pred")
                        pred_np = self.overlay(self.detected_glomus_list[file_key], 1, margin_x, margin_y, seg_pred_l, xmin, ymin, xmax, ymax, "pred")

                        # make ground truth and prediction result of WSI
                        whole_gt_np = self.generate_whole_img([xmin, ymin, xmax, ymax], whole_gt_np, gt_np)
                        whole_pred_np = self.generate_whole_img([xmin, ymin, xmax, ymax], whole_pred_np, pred_np)

                        iou_eval_val.addBatch(pred_np, gt_np)
                        self.iouEvalVal.addBatch(pred_np, gt_np)

                # save images
                output_gt_file_name = os.path.join(self.output_dir, file_key + "_gt.jpg")
                cv2.imwrite(output_gt_file_name, whole_gt_np)
                output_pred_file_name = os.path.join(self.output_dir, file_key + "_pred.jpg")
                cv2.imwrite(output_pred_file_name, whole_pred_np)

                overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, mDice = iou_eval_val.getMetricMicro()

                out_f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(file_key, overall_acc, per_class_acc,
                                                                  per_class_iou, mIOU,
                                                                  per_class_dice, mDice))
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(file_key, overall_acc, per_class_acc,
                                                          per_class_iou, mIOU,
                                                          per_class_dice, mDice))

            overall_acc, per_class_acc, per_class_iou, mIOU, per_class_dice, mDice = self.iouEvalVal.getMetricMicro()
            out_f.write("total\t{}\t{}\t{}\t{}\t{}\t{}".format(overall_acc, per_class_acc, per_class_iou, mIOU,
                                                               per_class_dice, mDice))


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
    parser.add_argument('--seg_pred_image_dir', help="Set path to the parent directory of the pred segmentation image files", type=str, required=True)
    parser.add_argument('--seg_gt_image_dir', dest='seg_gt_image_dir', help="Set path to parent directory of the segmentation grount truth json files. If you do not have the ground truth, no need to set this argument.", type=str, default=None)
    parser.add_argument('--object_detection_gt_xml_dir', dest='ob_gt_xml_dir', help="Set path to the parent directory of the segmentation object detection xml filesi. If you do not have the ground truth, no need to set this argument.", type=str,
                        default=None)
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
    model = Generate_Segmentation_Gt(args.staining, args.ob_gt_xml_dir, args.target_list, args.input_csv, args.iou_threshold, args.output_file, args.output_dir, args.wsi_dir, args.seg_gt_image_dir, args.window_size, args.seg_pred_image_dir, args.classes, args.no_save, args.start, args.end)
    model.read_detected_glomus_list()
    if args.seg_gt_image_dir is None or args.seg_pred_image_dir is None:
        # generate prediction wsi
        model.generate_pred_wsi()
    else:
        # generate prediction and gt wsi, and evaluate
        # model.scan_files()
        model.generate_pred_wsi()
