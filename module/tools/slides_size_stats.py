# -*- coding: utf-8 -*-

import argparse
import glob
import openslide
import os

def load_patient_id(id_list_file):
    patient_id_l = []
    for line in open(id_list_file):
        line = line.rstrip()
        patient_id_l.append(line)
    return patient_id_l

def read_slide(ndpi_file_path):
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
    slide = openslide.open_slide(ndpi_file_path[0])
    slide_x_y = (slide.dimensions)
    return slide_x_y

def write_file(patient_d, output_file):
    with open(output_file, "w") as out_f:
        for patient_id in patient_d.keys():
            out_f.write("{},{},{}\n".format(patient_id, patient_d[patient_id][0], patient_d[patient_id][1]))

def main(args):
    patient_id_l = load_patient_id(args.target_list)
    patient_d = {}
    for patient_id in patient_id_l:
        ndpi_l = glob.glob(os.path.join(args.wsi_dir, patient_id, "*ndpi"))
        print(ndpi_l)
        slide_x_y = read_slide(ndpi_l)
        patient_d[patient_id] = slide_x_y
    write_file(patient_d, args.output_file)
    
def parse_args():
    '''
    Abstract: 
        Parse input arguments
    return: 
        args
    '''
    parser = argparse.ArgumentParser(description='summarize slide sizes')
    parser.add_argument('--target_list', dest='target_list', help="Set path to the file of target list. In the file, ndpi filenames without extension are written. (txt file format)", type=str, required=True)
    parser.add_argument('--wsi_dir', dest='wsi_dir', help="Set path to parent directory of whole slide images", type=str, required=True)
    parser.add_argument('--output_file', dest='output_file', help="Set output file name", type=str, required = True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

