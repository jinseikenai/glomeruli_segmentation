# -*- coding: utf-8 -*-
# Copyright 2020 The University of Tokyo Hospital. All Rights Reserved.
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This program is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

from annotation_handler import AnnotationHandler
import argparse
from collections import OrderedDict
import csv
import glob
import json
from labelme import utils
import numpy as np
import os
import openslide
from PIL import Image, ImageDraw, ImageFont
import re
from utils import shape
from utils import my_lblsave
import xml.etree.ElementTree as ElementTree

MAGNIFICATION=8

def relabel(img):
    img[img == 13] = 4
    img[img == 12] = 3
    img[img == 11] = 2
    img[img == 8] = 1
    img[img == 7] = 0
    return img

class Generate_Segmentation_Gt(AnnotationHandler):
    """Evaluation and Visualization"""

    font = ImageFont.truetype('DejaVuSans.ttf', 10)

    def __init__(self, staining_type, annotation_dir, target_list, detect_list_file,
                 iou_threshold, output_dir, wsi_dir, gt_png_dir, seg_gt_json_dir, no_save=False, start=0, end=0):
        super(Generate_Segmentation_Gt, self).__init__(annotation_dir, staining_type)
        self.MARGIN = 20 # 20 micrometre 
        self.iou_threshold = iou_threshold
        self.detect_list_file = detect_list_file
        self.output_dir = output_dir
        self.image_ext = ['.PNG', '.png']
        self.detected_glomus_list = {}
        self.detected_patient_id = []
        self.image = None
        self.overlap_d = {} # overlap_list  #key: date, value: [{"gt":gt, "pred":found_rect, "iou": iou, "json": json_file_name_l[0]}]
        self.seg_gt_json_dir = seg_gt_json_dir
        self.wsi_dir = wsi_dir

        self.annotation_file_date_pattern = '^\d{8}_(.+)'
        self.re_annotation_file_date_pattern = re.compile(self.annotation_file_date_pattern)

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.glomus_category = ['glomerulus', 'glomerulus-kana']

        '''Flag indicating not saving visualization result.'''
        self.no_save = no_save

        self.target_list = target_list
        self.start = start
        self.end = end

    def scan_files(self):
        """
            Abstract: 
                1. Read ground truth of glomerular region.
                2. Read the glomeruli detection result file.
                3. Generate glomerular images and ground truth images for segmentation.
        """

        self.print_header()
        with open(self.target_list, "r") as list_file:
            lines = list_file.readlines()
            if self.end == 0 or self.end > len(lines):
                end = len(lines)
            else:
                end = self.end
            for i in range(self.start, end):
                [patient_id, file_body] = lines[i].split(os.sep)
                annotation_dir = os.path.join(self.annotation_dir, self.staining_dir)
                dir_path = os.path.join(annotation_dir, patient_id)
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

                                    recall, recall_hit_num = self.calculate_overlap_and_save_images(slide_name_body, int(body_list[0][2]))
                                    '''Ignore commas if file name has commas.'''
                                    self.print_result_record(body.replace(',', ''), recall, recall_hit_num,
                                                             str(len(self.gt_list)),
                                                             str(len(self.detected_glomus_list[slide_name_body])))


    def print_result_record(self, body, recall, recall_hit_num, num_gt, num_detected):
        print('"{}",{},{},{},{}'.format(body, recall, recall_hit_num, num_gt, num_detected))

    def print_header(self):
        print('data,recall,recall_hit_num,gt_num,detect_num')

    def save_image(self, path, file_name):
        if not self.no_save:
            self.image.save(os.path.join(path, file_name))

    def calculate_overlap_and_save_images(self, file_key, times):
        """
        Abstract: Calculate the overlap between pred and ground truth, and save ground truth images for segmenation
        Args:
            file_key: <str> directionary name. e.g. H16-09557
            times: <int>
        Return: 
            float(recall_hit_num) / float(gt_num): <float> recall
            recall_hit_num: <int> the number of detected glomerular region
        """
          

        '''Preparation for drawing result.'''
        if not self.no_save:
            '''Convert to RGBA to use transparent mode'''
            draw = ImageDraw.Draw(self.image, 'RGBA')
        else:
            draw = None

        '''Evaluate the recall'''
        gt_num = len(self.gt_list)
        recall_hit_num = 0
        index = -1
        overlap_l = [] # "org_gt": detection gt coordinate "margin_gt": segmentation gt coordinate including margi, "pred":predicted coordinate, "iou":overlap between gt and predicted bbox, "json": absolute path to the gt(json), "file_key": pathology number}
        detected_ind_l = []
        seg_gt_json_l = glob.glob(os.path.join(self.seg_gt_json_dir, file_key, "*.json"))
        print("file_key: {}".format(file_key))
        print("self.seg_gt_json_dir: {}".format(self.seg_gt_json_dir))
        ndpi_l = glob.glob(os.path.join(self.wsi_dir, file_key, "*ndpi"))
        assert len(ndpi_l) == 1
        ndpi_path_s = ndpi_l[0]
        margin_x, margin_y = self.read_slide_and_cal_margin(ndpi_path_s)
        output_org_dir = os.path.join(self.output_dir, "org_image", file_key)
        if not os.path.exists(output_org_dir):
            os.makedirs(output_org_dir)
        for gt in self.gt_list:
            # caluculate overlap between gt and predected bbox
            index += 1
            if self.gt_name_list[index] in self.glomus_category:
                if not self.no_save:
                    draw.rectangle([gt[0], gt[1], gt[2], gt[3]], fill=None, outline='yellow')
                # '''In the case of drawing label'''
                gt_l = list(map(lambda x: x * times, gt))
                gt_margin_l = []
                gt_margin_l.append(int(gt_l[0] - margin_x))
                gt_margin_l.append(int(gt_l[1] - margin_y))
                gt_margin_l.append(int(gt_l[2] + 2 * margin_x)) #extra right side due to bug of clip_annotated_area.py
                gt_margin_l.append(int(gt_l[3] + 2 * margin_y)) #extra right side due to bug of clip_annotated_area.py
                gt_margin_l = list(map(lambda x: x, gt_margin_l))
                for cor_i in gt_margin_l:
                    assert cor_i >= 0
                iou_list = []
                overlap_d = {}
                # search json file
                search_name = "xmin{}_ymin{}_xmax{}_ymax{}".format(int(gt_l[0]/MAGNIFICATION), int(gt_l[1]/MAGNIFICATION), int(gt_l[2]/MAGNIFICATION), int(gt_l[3]/MAGNIFICATION))
                json_file_name_l = [json_name for json_name in seg_gt_json_l if re.search(search_name, json_name)]
                assert len(json_file_name_l) <= 1
                if len(json_file_name_l) == 0:
                    # there is no annotation file(json). This is because the glomerulous is blurred, so, the glomerular image is not annotated
                    continue
                for ind, found_rect in enumerate(self.detected_glomus_list[file_key]):
                    iou = self.check_overlap(gt_l, found_rect)
                    if iou >= self.iou_threshold:
                        iou_list.append(iou)
                        if ("iou" in overlap_d.keys() and iou >= overlap_d["iou"]) or (not "iou" in overlap_d.keys()):
                            # TP
                            detected_ind = ind
                            overlap_d = {"org_gt":gt_l, "margin_gt":gt_margin_l, "pred":found_rect, "iou": iou, "json": json_file_name_l[0], "file_key": file_key}
                            # save org image
                            pred_region_pil = self.slide.read_region((found_rect[0], found_rect[1]), 0, (found_rect[2]-found_rect[0], found_rect[3]-found_rect[1]))
                            filename, _ = os.path.splitext(os.path.basename(json_file_name_l[0]))
                            output_file_name = "xmin{}_ymin{}_xmax{}_ymax{}".format(int(overlap_d["pred"][0]/MAGNIFICATION), int(overlap_d["pred"][1]/MAGNIFICATION), int(overlap_d["pred"][2]/MAGNIFICATION), int(overlap_d["pred"][3]/MAGNIFICATION))
                            overlap_d["name"] = output_file_name
                            pred_region_pil.save(os.path.join(output_org_dir, output_file_name + '.PNG'), format="PNG", quality=100)

                '''It may be overlapped with multiple detected rectangles.
                In such case, the rectangle with max IoU is regarded as overlapping with it.'''
                if len(iou_list) > 0:
                    assert "org_gt" in overlap_d.keys()
                    overlap_l.append(overlap_d)
                    recall_hit_num += 1
                    detected_ind_l.append(detected_ind)
                else:
                    # if this gt is not detected. i.e. FN
                    overlap_l.append({"org_gt":gt_l, "margin_gt": gt_margin_l, "pred":[], "iou":0, "json": json_file_name_l[0], "file_key": file_key})
        # FP
        ## extract index of FP
        wrongly_detected_ind_l = [x for x in range(len(self.detected_glomus_list[file_key])) if not x in detected_ind_l]
        for not_detected_ind in wrongly_detected_ind_l:
            xmin_not_detected = self.detected_glomus_list[file_key][not_detected_ind][0]
            ymin_not_detected = self.detected_glomus_list[file_key][not_detected_ind][1]
            xmax_not_detected = self.detected_glomus_list[file_key][not_detected_ind][2]
            ymax_not_detected = self.detected_glomus_list[file_key][not_detected_ind][3]
            output_file_name = "xmin{}_ymin{}_xmax{}_ymax{}".format(int(xmin_not_detected/MAGNIFICATION), int(ymin_not_detected/MAGNIFICATION), int(xmax_not_detected/MAGNIFICATION), int(ymax_not_detected/MAGNIFICATION))
            overlap_l.append({"org_gt":[], "margin_gt": [], "pred":self.detected_glomus_list[file_key][not_detected_ind], "iou":0, "json": "", "file_key": file_key, "name": output_file_name})
            # save org image
            print("FP:{}".format(self.detected_glomus_list[file_key][not_detected_ind]))
            pred_region_pil = self.slide.read_region((xmin_not_detected, ymin_not_detected), 0, (xmax_not_detected-xmin_not_detected, ymax_not_detected-ymin_not_detected))
            filename, _ = os.path.splitext(os.path.basename(json_file_name_l[0]))
            pred_region_pil.save(os.path.join(output_org_dir, output_file_name + '.PNG'), quality=100)
        self.overlap_d = {file_key:overlap_l}
        self.generate_org_gt_png()
        if not self.no_save:
            '''Evaluate the precision'''
            for found_rect in self.detected_glomus_list[file_key]:
                rect = list(map(lambda x: x / times, found_rect))
                draw.rectangle([rect[0], rect[1], rect[2], rect[3]], fill=None, outline='red')
                max_iou = 0.0
                for gt in self.gt_list:
                    iou = self.check_overlap(gt, rect)
                    if max_iou < iou:
                        max_iou = iou
                label = 'conf:{:.2f},IoU:{:.2f}'.format(found_rect[4], max_iou)
                (text_w, text_h) = draw.textsize(label)
                draw.rectangle([rect[0], rect[1] - text_h - 4,
                                rect[0] + text_w - 10, rect[1]], fill=(255, 0, 0, 128), outline=None)
                draw.text((rect[0] + 4, rect[1] - text_h - 2), label, fill=(255, 255, 255, 128),
                          font=Generate_Segmentation_Gt.font)

        if gt_num != 0:
            return float(recall_hit_num) / float(gt_num), recall_hit_num
        else:
            return 0, recall_hit_num

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
        """read target image"""
        for ext in self.image_ext:
            file_path = os.path.join(dir_path, body + ext)
            if os.path.isfile(file_path):
                self.image = Image.open(file_path)
                break

    def generate_org_gt_png(self):
        """generate original images and ground-truth images for segmentation"""
        od = OrderedDict()
        od['glomerulus'] = 1
        od['crescent'] =  2
        od['collapsing'] = 3
        od['sclerosis'] = 3
        od['mesangium'] = 4
        od['poler_mesangium'] = 4
        target_dic = {'all': od}
        for date, glomus_l in self.overlap_d.items():
            for ind, glomus_d in enumerate(glomus_l):
                output_label_dir = os.path.join(self.output_dir, "label", "all", glomus_d["file_key"])
                if not os.path.exists(output_label_dir):
                    os.makedirs(output_label_dir)
                # GT(binary) to numpy
                if glomus_d["json"] != "":
                    # TP/FN
                    data = json.load(open(glomus_d["json"]), object_pairs_hook=OrderedDict)
            
                    if data['imageData']:
                        imageData = data['imageData']
                    else:
                        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
                        with open(imagePath, 'rb') as f:
                            imageData = f.read()
                            imageData = base64.b64encode(imageData).decode('utf-8')
            
                    img = utils.img_b64_to_arr(imageData)
            
                    label_name_to_value = {'_background_': 0}
                    target_list = target_dic["all"]
                    for target in target_list:
                        label_name_to_value[target] = target_list[target]
                    label_name_list = target_list
                    lbl = shape.shapes_to_label(img.shape, data['shapes'], label_name_to_value, label_name_list)
                    # check the size
                    assert lbl.shape[1] == glomus_d["margin_gt"][2] - glomus_d["margin_gt"][0] # x-axis
                    assert lbl.shape[0] == glomus_d["margin_gt"][3] - glomus_d["margin_gt"][1] # y-axis
                    # make GT
                    if len(glomus_d["pred"]) > 0:
                        # TP
                        whole_area_xmin = min([glomus_d["pred"][0], glomus_d["margin_gt"][0]])
                        whole_area_ymin = min([glomus_d["pred"][1], glomus_d["margin_gt"][1]])
                        whole_area_xmax = max([glomus_d["pred"][2], glomus_d["margin_gt"][2]])
                        whole_area_ymax = max([glomus_d["pred"][3], glomus_d["margin_gt"][3]])
                        whole_area_np = np.zeros((int(whole_area_ymax - whole_area_ymin), int(whole_area_xmax - whole_area_xmin)))
                        # overay gt label on the union of predicted area and margin_gt
                        whole_area_np[glomus_d["margin_gt"][1]-whole_area_ymin:glomus_d["margin_gt"][3]-whole_area_ymin, glomus_d["margin_gt"][0]-whole_area_xmin:glomus_d["margin_gt"][2]-whole_area_xmin] = lbl
                        # crop predected area
                        lbl_pred = whole_area_np[glomus_d["pred"][1]-whole_area_ymin:glomus_d["pred"][3]-whole_area_ymin, glomus_d["pred"][0]-whole_area_xmin:glomus_d["pred"][2]-whole_area_xmin]
                        # save ground truth of detected area
                        my_lblsave.lblsave(os.path.join(output_label_dir, glomus_d["name"] + '.PNG'), lbl_pred)
                    else:
                        # FN
                        # no need to draw FP images
                        pass
          
                else:
                    # FP. Outputs background image as ground truth to evaluate segmentation on WSI
                    if len(glomus_d["pred"]) > 0:
                        if "name" in glomus_d:
                            whole_area_np = np.zeros((int(glomus_d["pred"][3]-glomus_d["pred"][1]), int(glomus_d["pred"][2]-glomus_d["pred"][0])))
                            my_lblsave.lblsave(os.path.join(output_label_dir, glomus_d["name"] + '.PNG'), whole_area_np)
                    else:
                        # there is no annotation file(json). This is because the glomerulous is blurred, so, the glomerular image is not annotated
                        pass
                
    def read_slide_and_cal_margin(self, ndpi_file_path):
        """calculate margin for segmentation"""
        self.slide = openslide.open_slide(ndpi_file_path)
        mpp_x = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        margin_x = int(round(float(self.MARGIN) / mpp_x))
        margin_y = int(round(float(self.MARGIN) / mpp_y))
        return margin_x, margin_y

    def output_org_files(self):
        """output original files."""
        for file_key in self.detected_glomus_list.keys():
            ndpi_l = glob.glob(os.path.join(self.wsi_dir, file_key, "*ndpi"))
            assert len(ndpi_l) == 1
            ndpi_path_s = ndpi_l[0]
            margin_x, margin_y = self.read_slide_and_cal_margin(ndpi_path_s)
            output_org_dir = os.path.join(self.output_dir, "org_image", file_key)
            if not os.path.exists(output_org_dir):
                os.makedirs(output_org_dir)
            for ind, found_rect in enumerate(self.detected_glomus_list[file_key]):
                # save org image
                pred_region_pil = self.slide.read_region((found_rect[0], found_rect[1]), 0, (found_rect[2]-found_rect[0], found_rect[3]-found_rect[1]))
                output_file_name = "xmin{}_ymin{}_xmax{}_ymax{}".format(int(found_rect[0]/MAGNIFICATION), int(found_rect[1]/MAGNIFICATION), int(found_rect[2]/MAGNIFICATION), int(found_rect[3]/MAGNIFICATION))
                pred_region_pil.save(os.path.join(output_org_dir, output_file_name + '.PNG'), format="PNG", quality=100)

def parse_args():
    """
        Abstract: Parse input arguments
        Return: args
    """
    parser = argparse.ArgumentParser(description='Make segmentation data from the result of the detection')
    parser.add_argument('--staining', dest='staining', help="Set --staining for staining method. e.g. OPT_PAS", type=str, required=True)
    parser.add_argument('--merged_detection_result_csv', dest='input_csv', help="Set path to the merged detection result (csv file format)", type=str, required=True)
    parser.add_argument('--target_list', dest='target_list', help="Set path to the file of target list. In the file, ndpi filenames without extension are written. (txt file format)", type=str, required=True)
    parser.add_argument('--wsi_dir', dest='wsi_dir', help="Set path to the parent directory of whole slide images", type=str, required=True)
    parser.add_argument('--segmentation_gt_json_dir', dest='seg_gt_json_dir', help="Set path to parent directory of the segmentation grount truth json files. If you do not have the ground truth, no need to set this argument.", type=str, default=None)
    parser.add_argument('--object_detection_gt_xml_dir', dest='ob_gt_xml_dir', help="Set path to the parent directory of the segmentation object detection xml files. If you do not have the ground truth, no need to set this argument.", type=str, default=None)
    parser.add_argument('--iou_threshold', dest='iou_threshold', help="Set iou threshold", type=float, default=0.01)
    parser.add_argument('--output_dir', dest='output_dir', help="Set path to the output directory. If you do not set this argument, output files to ./output/seg_data.", type=str, default='./output/seg_data')
    parser.add_argument('--start', dest='start', help="Set --start for start line begin 0", type=int, default=0)
    parser.add_argument('--end', dest='end', help="Set --end for end line", type=int, default=0)
    parser.add_argument('--segmentation_gt_png_dir', dest='gt_png_dir', help="Set path to parent directory of the label images. If you do not have the ground truth, no need to set this argument.", type=str, default=None)
    parser.add_argument('--no_save', dest='no_save', help="set --no_save for test", action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = Generate_Segmentation_Gt(args.staining, args.ob_gt_xml_dir, args.target_list, args.input_csv, args.iou_threshold, args.output_dir, args.wsi_dir, args.gt_png_dir, args.seg_gt_json_dir, args.no_save, args.start, args.end)
    model.read_detected_glomus_list()
    if args.seg_gt_json_dir is None or args.ob_gt_xml_dir is None:
        model.output_org_files()
    else:
        assert args.ob_gt_xml_dir is not None
        model.scan_files()
