# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

def plot(args):
    df = pd.read_csv(args.loss_tsv, header=0, index_col=0, delimiter="\t")
    print(df.columns)
    ax = df[["Loss (train)", "Loss (val)", "mIoU (train)","mIoU (val)"]].plot(secondary_y=["mIoU (train)","mIoU (val)"], mark_right=False)
    ax.set_ylabel('Loss', fontsize=15)
    ax.right_ax.set_ylabel('mIoU', fontsize=15)
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylim(0, 1)
    ax.right_ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    plt.savefig(args.output_png)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--loss_tsv', required=True, help='Set path to loss file generated during training.')
    parser.add_argument('--output_png', required=True, help='Set path to output png file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    assert ".png" in args.output_png
    plot(args)
