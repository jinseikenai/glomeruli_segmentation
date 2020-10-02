# -*- coding: utf-8 -*-

import csv
import glob
import numpy as np
from PIL import Image as PILImage
from argparse import ArgumentParser
import os

def main(args):
    file_l = file_list(args)
    data_summary_l = []
    for file_name in file_l:
        data_summary_l.append(load_data(args, file_name))
    write_csv(args, data_summary_l)

def img_relabel(img):
    if img.flags.writeable == False:
        img_cp = img.copy()
        img = img_cp
    img[img == 13] = 4
    img[img == 12] = 3
    img[img == 11] = 2
    img[img == 8] = 1
    img[img == 7] = 0
    return img

def file_list(args):
    file_l = glob.glob(os.path.join(args.label_data_dir, "H*", "*.{}".format(args.img_extn)))
    return file_l

def load_data(args, file_name):
    file_name_split = file_name.split("/")
    assert "H" in file_name_split[-2]
    patient_id = file_name_split[-2]
    xmin, ymin, xmax, ymax = extract_cor(args, file_name_split[-1])
    img_label = PILImage.open(file_name)
    img_label = np.asarray(img_label)
    if args.data_type == "pred":
        img_label = img_relabel(img_label)
    background_px = np.count_nonzero(img_label==0)
    glomeruli_px = np.count_nonzero(img_label==1)
    crescent_px = np.count_nonzero(img_label==2)
    sclerosis_px = np.count_nonzero(img_label==3)
    mesangium_px = np.count_nonzero(img_label==4)
    assert background_px > 0
    #data_summary_d = {"patient_id": patient_id, "file_name": file_name_split[-1], "background": background_px, "glomeruli_px": glomeruli_px, "crescent_px": crescent_px, "sclerosis_px": sclerosis_px, "mesangium_px": mesangium_px}
    data_summary_l = [patient_id, file_name_split[-1], xmin, ymin, xmax, ymax, background_px, glomeruli_px, crescent_px, sclerosis_px, mesangium_px]
    return data_summary_l

def extract_cor(args, string):
    string_split = string.split("_")
    for split in string_split:
        if "xmin" in split:
            xmin = split.lstrip("xmin")
        elif "ymin" in split:
            ymin = split.lstrip("ymin")
        elif "xmax" in split:
            xmax = split.lstrip("xmax")
        elif "ymax" in split:
            ymax = split.lstrip("ymax")
            ymax = ymax.rstrip(".{}".format(args.img_extn))
    return xmin, ymin, xmax, ymax

def write_csv(args, data_summary_l):
    with open(args.output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'file_name','xmin', 'ymin', 'xmax', 'ymax', 'background', 'glomerulus', 'crescent', 'sclerosis', 'mesangium'])
        for data_summary in data_summary_l:
            writer.writerow(data_summary)

def parse_args():
    parser = ArgumentParser(description='Glomerular segmentation on the cropped images')
    parser.add_argument('--label_data_dir', required=True, help='Set path to parent directory of label images')
    parser.add_argument('--img_extn', default="PNG", help='Set image extinction')
    parser.add_argument('--data_type', default="ground-truth", choices=["pred", "ground-truth"], help='Set data type (pred or ground-truth)')
    parser.add_argument('--output_csv', default='./result.csv', help='Set path to the output csv name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    assert "csv" in args.output_csv
    main(args)
