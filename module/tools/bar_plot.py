# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

def main(pred_csv, gt_csv, output_png, graph_type, output_csv):
    pred_df = pd.read_csv(pred_csv, header=0, delimiter=",")
    gt_df = pd.read_csv(gt_csv, header=0, delimiter=",")
    print(pred_df)
    patient_id_np = pred_df["patient_id"].unique()
    print(patient_id_np.shape[0])
    if patient_id_np.shape[0] > 8:
        x = 5
    else:
        x = 4
    fig, ax = plt.subplots(2, x, sharex='col', sharey='row')
    sum_fig_l = []
    rate_fig_l = []
    cols_l = ["Prediction", "Ground truth"]
    df = pd.DataFrame(index=["glomerulus", "crescent", "sclerosis", "mesangium"], columns=[])
    for ind, patient_id_str in enumerate(patient_id_np):
        pred_ex_pd = pred_df[pred_df["patient_id"]==patient_id_str]
        gt_ex_pd = gt_df[gt_df["patient_id"]==patient_id_str]
        # calcurate summation
        if graph_type == "sum":
            sum_pred_pd = sum_pix(pred_ex_pd)
            sum_gt_pd = sum_pix(gt_ex_pd)
            merged_ex_pd = pd.concat([sum_pred_pd, sum_gt_pd], axis=1)
            merged_ex_pd.columns = cols_l
            sum_fig_l.append(draw_bars(merged_ex_pd, output_png, ind, ax, x, 2500))
        elif graph_type == "rate":
            # calcurate rate
            pred_rate_mean_pd, gt_rate_mean_pd = rate_pix(pred_ex_pd, gt_ex_pd)
            merged_rate_pd = pd.concat([pred_rate_mean_pd, gt_rate_mean_pd], axis=1)
            merged_rate_pd.columns = cols_l
            rate_fig_l.append(draw_bars(merged_rate_pd, output_png, ind, ax, x, 1))
            df = pd.concat([df, merged_rate_pd[cols_l[0]] - merged_rate_pd[cols_l[1]]], axis=1)
    df = df.apply(lambda x:abs(x))
    df.to_csv(output_csv)
    if graph_type == "sum":
        fig.legend(sum_fig_l, labels=["Prediction", "Ground truth"])
        plt.gcf().text(0.005,0.6,"μm$^{2}$",rotation=90)
    elif graph_type == "rate":
        fig.legend(rate_fig_l, labels=["Prediction", "Ground truth"])
        plt.gcf().text(0.005,0.5,"Average rate",rotation=90)
    plt.gcf().text(0.5,0.005,"class")
    plt.tight_layout()
    fig.savefig(output_png)
    

def sum_pix(org_pd):
    sum_pd = org_pd.loc[:,["glomerulus", "crescent", "sclerosis", "mesangium"]].sum()
    sum_um_pd = np.sqrt(sum_pd * 0.23)
    return sum_um_pd

def rate_pix(pred_pd, gt_pd):
    pred_rate_pd = pred_pd.loc[:,["glomerulus", "crescent", "sclerosis", "mesangium"]].apply(lambda x:x/sum(x), axis=1)
    pred_rate_mean_pd = pred_rate_pd.mean()
    gt_rate_pd = gt_pd.loc[:,["glomerulus", "crescent", "sclerosis", "mesangium"]].apply(lambda x:x/sum(x), axis=1)
    gt_rate_mean_pd = gt_rate_pd.mean()
    return pred_rate_mean_pd, gt_rate_mean_pd

def draw_bars(merged_pd, output_png, ind, ax, xsize, ymax):
    print(ind)
    x = 0 if ind < xsize else 1
    y = ind % xsize
    ax[x,y].set_ylim(0, ymax)
    l = merged_pd.plot(ax=ax[x, y], kind='bar', legend=False)
    plt.subplots_adjust(left=0.15)
    return l
   
def draw_bar(merged_pd, output_png, ind, ax):
    print(ind)
    x = 0 if ind < 5 else 1
    y = ind % 5
    ax[x,y].set_ylim(0, 2500)
    merged_pd.plot(ax=ax[x, y], kind='bar')
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("μm$^{2}$")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pixel_pred_csv', required=True, help='Set path to pixel info file.')
    parser.add_argument('--pixel_gt_csv', required=True, help='Set path to pixel info (gt) file.')
    parser.add_argument('--output_png', required=True, help='Set path to output png file')
    parser.add_argument('--output_summary_csv', required=True, help='Set path to output summary file')
    parser.add_argument('--graph_type', choices=["sum", "rate"], required=True, help='Set graph type')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    assert ".png" in args.output_png
    main(args.pixel_pred_csv, args.pixel_gt_csv, args.output_png, args.graph_type, args.output_summary_csv) 
