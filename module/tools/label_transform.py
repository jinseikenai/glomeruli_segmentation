# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import numpy as np
import cv2
import glob
from PIL import Image as PILImage

pallete = [[0, 0, 0],
           [255, 0, 0],
           [0,184, 0],
           [255, 255, 0],
           [0,0,255],
           [128, 64, 128],
           [244, 35, 232],
           [255, 128, 0],
           [0, 128, 0],
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

def transform(args):
    file_l = glob.glob("{}/*/*.PNG".format(args.parent_dir))
    print(file_l)
    for filename in file_l:
        print("Filename:{}".format(filename))
        img_pil = PILImage.open(filename)
        pc_list = img_pil.getpalette()
        img_np = np.asarray(img_pil)
        print("Num of mesangium pixels:{}".format(np.count_nonzero(img_np==4)))
        img_tran_np = np.where(img_np == 4, 1, img_np)
        with PILImage.fromarray(img_tran_np, mode="P") as img:
            img.putpalette(pc_list)
            img.save(filename)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--parent_dir', required=True, help='Set path to parent file.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    transform(args)
