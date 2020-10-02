# -*- coding: utf-8 -*-
import argparse
import os
import glob

def main():
    Create_txt(train_rgb_dir, train_label_dir, train_txt)
    Create_txt(val_rgb_dir, val_label_dir, val_txt)

def Create_txt(rgb_dir, label_dir, txt_path):
    filelist_rgb = glob.glob(os.path.join(rgb_dir, '**/*.PNG'), recursive=True)
    txt = os.path.basename(txt_path)
    try:
        with open(txt_path, mode='w') as f:
            for file_path_rgb in filelist_rgb:
                filename = file_path_rgb.split("/")
                file_path_label = os.path.join(label_dir, filename[-2], filename[-1])
                if os.path.exists(file_path_rgb):
                    s = file_path_rgb + ',' + file_path_label + '\n'
                    f.write(s)
                else:
                    print("{] does not exists.".format(file_path_rgb))
                    print("{] does not exists.".format(file_path_label))

    except FileExistsError:
        print(txt, 'already exists')

def parse_args():
    parser = argparse.ArgumentParser(
        description='This program makes trainval list')
    parser.add_argument('--data_dir', type=str,
                        required=True, help="Set path to parent data directory")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    TRAINVAL = args.data_dir
    train_rgb_dir = TRAINVAL + '/train/rgb'
    train_label_dir = TRAINVAL + '/train/label'
    val_rgb_dir = TRAINVAL + '/val/rgb'
    val_label_dir = TRAINVAL + '/val/label'

    train_txt = TRAINVAL + '/train.txt'
    val_txt = TRAINVAL + '/val.txt'

    main()
