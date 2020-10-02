# -*- coding: utf-8 -*-

import argparse
import csv
import os
import glob

def parse_args():
    '''
    Parse input arguments
    :return: args
    '''
    parser = argparse.ArgumentParser(description='MERGE_OVERLAPPED_GLOMUS')
    parser.add_argument('--base_list_csv', dest='base_list_csv', help="Please set path to the base_list.csv", type=str)
    parser.add_argument('--data_dir', dest='data_dir', help="Please set path to the data dir", type=str)
    parser.add_argument('--output_file', dest='output_file', help="Please set path to the output target_list.txt", type=str)
    return parser.parse_args()


def make_list(args):
    with open(args.base_list_csv, "r") as csv_file:
        base_list_csv = csv.reader(csv_file)
        wsi_dir_tmp_l = []
        for row in base_list_csv:
            print(row)
            wsi_dir_tmp_l.append(row[3])
        wsi_dir_l = set(wsi_dir_tmp_l)
        print(wsi_dir_l)
        with open(args.output_file, "w") as out_f:
            for wsi_dir_name in wsi_dir_l:
                wsi_png_name_l = glob.glob(os.path.join(args.data_dir, wsi_dir_name, "*ndpi"))
                print(wsi_png_name_l)
                assert len(wsi_png_name_l) == 1
                png_name = wsi_png_name_l[0].split("/")[-1]
                output_name = "{}/{}".format(wsi_dir_name, png_name)
                out_f.write(os.path.splitext(output_name)[0] + "\n")
        

if __name__ == "__main__":
    args = parse_args()
    make_list(args)


