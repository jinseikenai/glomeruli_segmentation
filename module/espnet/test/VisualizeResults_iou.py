# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
from collections import defaultdict
import json
from PIL import Image as PILImage
import Model as Net
import os
from argparse import ArgumentParser
from IOUEval import iouEval
from labelme import utils
from boundary_extractor import bound2line

# modified https://github.com/sacmehta/ESPNet/blob/master/test/Visuazlize.py by Issei Nakamura

pallete = [[0, 0, 0],
           [255, 0, 0], 
           [0,184, 0], 
           [255, 255, 0], 
           [0,0,255], 
           [128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
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


label_idx = {
    1: "glomerulus",
    2: "crescent",
    3: "sclerosis",
    4: "mesangium",
}

def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def evaluateModel(args, model, up, rgb_image_list, label_image_list, device):
    # gloabl mean and std values (BGR)
    mean = list(map(float, args.mean))
    std = list(map(float, args.std))
    width = args.inWidth
    height = args.inHeight
    print("num of image:{}".format(len(rgb_image_list)))
    iouEvalVal = iouEval(args.classes)
    save_summary_acc = os.path.join(args.savedir, "summary_accuracy.csv")
    save_summary_data = os.path.join(args.savedir, "summary_dataset.csv")
    save_summary_pixel = os.path.join(args.savedir, "summary_pixel.csv")
    dataset_d = defaultdict(lambda :defaultdict(int))
    with open(save_summary_acc, "w") as summary_acc, open(save_summary_data, "w") as summary_data, open(save_summary_pixel, "w") as summary_pixel:
        summary_acc.write("filename,glomerulus, crescent, sclerosis, mesangium, background iou,glomerulus iou,crescent iou,sclerosis iou, mesangium iou,mIoU\n")
        summary_data.write("patient_id, glomerulus, crescent, sclerosis, mesangium\n")
        summary_pixel.write("patient_id, filename, background, glomerulus, crescent, sclerosis, mesangium\n")
        for i, (imgName, labelName) in enumerate(zip(rgb_image_list, label_image_list)):
            print("imgName: {}".format(imgName))
            patient_id = os.path.basename(os.path.dirname(imgName))
            img = cv2.imread(imgName)
            # if args.overlay:
            img_orig = np.copy(img)

            img = img.astype(np.float32)
            for j in range(3):
                img[:, :, j] -= mean[j]
            for j in range(3):
                img[:, :, j] /= std[j]

            # resize the image to 1024x512x3
            img = cv2.resize(img, (width, height))

            img /= 255
            img = img.transpose((2, 0, 1)) # convert to RGB
            img_tensor = torch.from_numpy(img)
            img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
            img_variable = Variable(img_tensor, volatile=True)
            if args.gpu_id >= 0:
                img_variable = img_variable.to(device)
            img_out = model(img_variable)

            if args.modelType == 2:
                img_out = up(img_out)

            classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
            classMap_numpy = cv2.resize(classMap_numpy,(img_orig.shape[1], img_orig.shape[0]),interpolation=cv2.INTER_NEAREST)
            if i % 100 == 0:
                print(i)

            name = imgName.split('/')[-1]
            name_rsplit = name.rsplit(".", 1)
            output_dir = os.path.join(args.savedir, patient_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                os.makedirs(os.path.join(output_dir, 'img'))
                os.makedirs(os.path.join(output_dir, 'json'))

            if args.colored:
                classMap_numpy_color = np.zeros((img_orig.shape[0], img_orig.shape[1], img_orig.shape[2]), dtype=np.uint8)
                for idx in range(len(pallete)):
                    [r, g, b] = pallete[idx]
                    classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
                name_org = name_rsplit[0] + ".png"
                cv2.imwrite(output_dir + os.sep + 'img' + os.sep + name_org, img_orig)
                if args.overlay:
                    overlayed = cv2.addWeighted(img_orig, 0.4, classMap_numpy_color, 0.6, 0)
                    name_over = name_rsplit[0] + "_overlay" + ".jpg"
                    cv2.imwrite(output_dir + os.sep + 'img' + os.sep + name_over, overlayed)
            background_px = np.count_nonzero(classMap_numpy==0)
            glomeruli_px = np.count_nonzero(classMap_numpy==1)
            crescent_px = np.count_nonzero(classMap_numpy==2)
            sclerosis_px = np.count_nonzero(classMap_numpy==3)
            mesangium_px = np.count_nonzero(classMap_numpy==4)
            summary_pixel.write("{},{},{},{},{},{},{}\n".format(patient_id, name.replace(args.img_extn, 'png'), background_px, glomeruli_px, crescent_px, sclerosis_px, mesangium_px))

            # セグメンテーション結果の境界線を抽出する
            boundary_lines = bound2line(classMap_numpy, max_classes=4)

            if args.cityFormat:
                classMap_numpy = relabel(classMap_numpy.astype(np.uint8))
            # save json file
            output_d = {}
            output_d["shapes"] = []
            for idx, label in label_idx.items():
                if idx in boundary_lines and len(boundary_lines[idx]) > 0:
                    for i in range(len(boundary_lines[idx])):
                        b_obj = {
                            "line_color": None,
                            "points": boundary_lines[idx][i].tolist(),
                            "fill_color": None,
                            "label": label,
                        }
                        output_d["shapes"].append(b_obj)
            output_d["lineColor"] = [0, 0, 0, 255]
            name_org = name_rsplit[0] + ".png"
            output_d["imagePath"] = '../img/' + name_org
            output_d["flags"] = {}
            output_d["fillColor"] = [0, 0, 0, 255]
            # output_d["imageData"] = utils.img_arr_to_b64(classMap_numpy).decode('utf-8')
            output_d["imageData"] = utils.img_arr_to_b64(img_orig).decode('utf-8')
            output_json_file = os.path.join(output_dir, 'json', name.replace(args.img_extn, 'json'))
            with open(output_json_file,'w') as out_json:
                json.dump(output_d, out_json, indent=4)
            # save org img
            output_png_file = os.path.join(output_dir, name.replace(args.img_extn, 'png'))
            # evaluate and generate combined images including original, prediction, ground-truth
            # compute the confusion matrix
            print("labelName: {}".format(labelName))
            if labelName is not None:
                assert os.path.basename(imgName) == os.path.basename(labelName)
                img_label = PILImage.open(labelName)
                img_label = np.asarray(img_label)
                # check original image and label image size
                assert img_label.shape[0] == img_orig.shape[0]
                assert img_label.shape[1] == img_orig.shape[1]
                img_label_re = cv2.resize(img_label, (width, height), interpolation=cv2.INTER_NEAREST)
                unique_values = np.unique(img_label_re)
                img_label_tensor = torch.from_numpy(img_label_re)
                img_label_tensor = torch.unsqueeze(img_label_tensor, 0)  # add a batch dimension
                for i in unique_values.tolist():
                    dataset_d[patient_id][i] += 1
                eachiouEvalVal = iouEval(args.classes)
                _ = iouEvalVal.addBatch(img_out.max(1)[1].data, img_label_tensor)
                hist = eachiouEvalVal.addBatch(img_out.max(1)[1].data, img_label_tensor)
                overall_acc, per_class_acc, per_class_iou, _ = eachiouEvalVal.getMetricRight()
                # write summary
                hist_tp_fn_fp = hist.sum(1) + hist.sum(0) - np.diag(hist)
                per_class_iou_ex = np.diag(hist)[hist_tp_fn_fp > 0.]/hist_tp_fn_fp[hist_tp_fn_fp > 0.]
                per_class_iou_ex = np.diag(hist)[unique_values]/hist_tp_fn_fp[unique_values]
                mIoU_each = np.nanmean(per_class_iou_ex)
                glomeruli = 1 if np.count_nonzero(unique_values==1) else 0
                crescent = 1 if np.count_nonzero(unique_values==2) else 0
                sclerosis = 1 if np.count_nonzero(unique_values==3) else 0
                mesangium = 1 if np.count_nonzero(unique_values==4) else 0
                summary_acc.write("{}/{},{},{},{},{},{},{},{},{},{},{}\n".format(patient_id, name.replace(args.img_extn, 'png'),glomeruli, crescent, sclerosis, mesangium, per_class_iou[0],per_class_iou[1],per_class_iou[2], per_class_iou[3], per_class_iou[4], mIoU_each))
                # generate combined image including original, prediction, ground-truth
                org_height = img_orig.shape[0]
                org_width = img_orig.shape[1]
                classMap_gt_np = np.zeros((img_orig.shape[0], img_orig.shape[1], img_orig.shape[2]), dtype=np.uint8)
                for idx in range(len(pallete)):
                    [r, g, b] = pallete[idx]
                    classMap_gt_np[img_label == idx] = [b, g, r]
                overlayed_gt = cv2.addWeighted(img_orig, 0.4, classMap_gt_np, 0.6, 0)
                combined_np = np.zeros((org_height, org_width*3, 3), dtype=int)
                combined_np[0:org_height, 0:org_width,:] = img_orig
                combined_np[0:org_height, org_width:2*org_width,:] = overlayed_gt
                combined_np[0:org_height, 2*org_width:3*org_width,:] = overlayed
                output_3_dir = os.path.join(args.savedir, "combined_images", patient_id)
                if not os.path.exists(output_3_dir):
                    os.makedirs(output_3_dir)
                output_3_png_file = os.path.join(output_3_dir, name.replace(args.img_extn, 'png'))
                cv2.imwrite(output_3_png_file, combined_np)
        if label_image_list[0] is not None:
            for patient, values_d in dataset_d.items():
                summary_data.write(patient)
                for i in range(1, args.classes):
                    summary_data.write(",{}".format(values_d[i]))
                summary_data.write("\n")
            overall_acc, per_class_acc, per_class_iou, mIOU = iouEvalVal.getMetricRight()
            overall_accuracy_output = os.path.join(args.savedir, "overall_accuracy.txt")
            with open(overall_accuracy_output, "w") as overall_accuracy_output_file:
                overall_accuracy_output_file.write("overall_acc:{}, per_class_acc:{}, per_class_iou:{}, mIOU:{}".format(overall_acc, per_class_acc, per_class_iou, mIOU))


def main(args):
    # read all the images in the folder
    rgb_image_list = sorted(glob.glob(args.rgb_data_dir + "/" + '*' + "/" + '*.PNG' ))
    if args.label_data_dir is not None:
        label_image_list = sorted(glob.glob(args.label_data_dir + "/" + '*' + "/" + '*.PNG' ))
        assert len(rgb_image_list) == len(label_image_list)
    else:
        label_image_list = [None]*len(rgb_image_list)
    if args.gpu_id >= 0:
        device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    print(device)

    up = None
    if args.modelType == 2:
        up = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        up = up.to(device)

    p = args.p
    q = args.q
    classes = args.classes
    if args.modelType == 2:
        modelA = Net.ESPNet_Encoder(classes, p, q) 
        model_weight_file = args.weights
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/encoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file, map_location=device))
    elif args.modelType == 1:
        modelA = Net.ESPNet(classes, p, q) 
        model_weight_file = args.weights
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file, map_location=device))
    else:
        print('Model not supported')
    modelA = modelA.to(device)
    # set to evaluation mode
    modelA.eval()

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    evaluateModel(args, modelA, up, rgb_image_list, label_image_list, device)


if __name__ == '__main__':
    parser = ArgumentParser(description='Glomerular segmentation on the cropped images')
    parser.add_argument('--rgb_data_dir', default="./data", required=True, help='Set path to parent directory of original glomerular images')
    parser.add_argument('--label_data_dir', default=None, help='Set path to parent directory of label images if you want to evaluate accuracy')
    parser.add_argument('--img_extn', default="PNG", help='Set image extinction')
    parser.add_argument('--inWidth', type=int, default=1024, help='Set width of resizing')
    parser.add_argument('--inHeight', type=int, default=512, help='Set height of resizing')
    parser.add_argument('--scaleIn', type=int, default=1, help='Set scale parameter. For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--modelType', type=int, default=1, help='Set model type. 1=ESPNet, 2=ESPNet-C')
    parser.add_argument('--savedir', default='./results', help='Set path to the output directory of the results')
    parser.add_argument('--gpu_id', default=-1, type=int, help='Set gpu id. If -1, then CPU.')
    parser.add_argument('--decoder', action='store_true', help='Set True if ESPNet. False for ESPNet-C')  # False for encoder
    parser.add_argument('--weights', required=True, help='Set path to the weights file')
    parser.add_argument('--mean', required=True, nargs='*', help='Set gloabal mean values (BGR) of training images')
    parser.add_argument('--std', required=True, nargs='*', help='Set global standard deviation values (BGR) of training images')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier. Supported only 2')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier. Supported only 3, 5, 8')
    parser.add_argument('--cityFormat', action='store_true', help='Set if you want to convert to cityscape '
                                                                       'original label ids')
    parser.add_argument('--colored', action='store_true', help='Set if you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--overlay', action='store_true', help='Set if you want to visualize the '
                                                                   'segmentation masks overlayed on top of RGB image')
    parser.add_argument('--classes', default=5, type=int, help='Set number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    print(args.decoder)
    if args.overlay:
        args.colored = True # This has to be true if you want to overlay
    main(args)
