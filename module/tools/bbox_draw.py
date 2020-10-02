# -*- coding: utf-8 -*-
from PIL import Image as Image
from PIL import ImageDraw
from argparse import ArgumentParser
import xml.etree.ElementTree as ElementTree
import openslide
import glob
import os

def parse_args():
    parser = ArgumentParser(description='Depict Glomerular area')
    parser.add_argument('--raw_image', type=str, help='Set path to raw image')
    parser.add_argument('--ndpi_image',  type=str, help='Set path to ndpi image')
    parser.add_argument('--annotation_file', type=str, help='Set path to annotation file (xml)')
    parser.add_argument('--output_image', type=str, help='Set path to the output image name')
    parser.add_argument('--output_dir', type=str, help='Set path to the output image directory')
    parser.add_argument('--width', default=10, type=int, help='rectangle line width')
    parser.add_argument('--wsi_dir', default=None, help='Set path to wsi directory if the argument "target_list" is set')
    parser.add_argument('--target_list', dest='target_list', help="Set path to the file of target list. In the file, ndpi filenames without extension are written. (txt file format)", type=str)
    args = parser.parse_args()
    return args

def load_patient_id(id_list_file):
    patient_id_l = []
    for line in open(id_list_file):
        line = line.rstrip()
        patient_id_l.append(line)
    return patient_id_l


def load_xml(xml_file):
    gt_list = []
    tree = ElementTree.parse(xml_file)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        if bbox != None:
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            gt_list.append([x1, y1, x2, y2])
    return gt_list

def draw(pil_image, output_image, gt_list, width, margin_x=0, margin_y=0):
    draw = ImageDraw.Draw(pil_image)
    for box in gt_list:
        draw.rectangle(((box[0]-margin_x, box[1]-margin_y), (box[2] + 2* margin_x, box[3] + 2 * margin_y)), fill=None, outline='yellow', width=width)
    pil_image.save(output_image)

def read_slide_and_cal_margin(ndpi_file_path):
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
    slide = openslide.open_slide(ndpi_file_path)
    MARGIN = 20
    #slide_width, slide_height = self.slide.dimensions
    mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
    margin_x = int(round(float(MARGIN) / mpp_x))/8
    margin_y = int(round(float(MARGIN) / mpp_y))/8
    print(slide.level_dimensions)
    return margin_x, margin_y


def main(args):
    file_list = [] # raw_image, ndpi, xml, output
    if args.wsi_dir is not None:
        patient_id_l = load_patient_id(args.target_list)
        for patient_id in patient_id_l:
            ndpi_l = glob.glob(os.path.join(args.wsi_dir, patient_id, "*ndpi"))
            gt_l = glob.glob(os.path.join(args.wsi_dir, patient_id, 'annotations', "*xml"))
            png_l = glob.glob(os.path.join(args.wsi_dir, patient_id, '*PNG'))
            output_dir = os.path.join(args.output_dir, patient_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_list.append([png_l[0], ndpi_l[0], gt_l[0], os.path.join(output_dir, "overlay_linewidth{}.PNG".format(args.width))])
    else:
        file_list.append([args.raw_image, args.ndpi_image,args.annotation_file, args.output_image])
    for files_l in file_list:
        margin_x, margin_y = read_slide_and_cal_margin(files_l[1])
        gt_list = load_xml(files_l[2])
        pil_image = Image.open(files_l[0])
        #draw(pil_image, files_l[3], gt_list, args.width, margin_x, margin_y)
        draw(pil_image, files_l[3], gt_list, args.width, 0, 0)

if __name__ == '__main__':
    args = parse_args()
    if args.raw_image != None:
        assert args.raw_image != args.output_image
    main(args)
