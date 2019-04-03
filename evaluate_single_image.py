# -*- coding:utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import bfscore
import math

major = cv2.__version__.split('.')[0]     # Get opencv version
bDebug = False


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(img_path, gt_path, num_classes=18):
    hist = np.zeros((num_classes, num_classes))

    try:
        label = Image.open(gt_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape
        imgsz = image_array.shape

        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        hist += fast_hist(label_array, image_array, num_classes)
    except Exception as err:
        print(err)

    return hist


def show_result(hist, n_cl=18):
    # Dressup 10K, 18 classes
    classes = ['background', 'hat', 'hair', 'sunglasses', 'upperclothes', 'skirt', 'pants', 'dress',
               'belt', 'leftShoe', 'rightShoe', 'face', 'leftLeg', 'rightLeg', 'leftArm', 'rightArm', 'bag', 'scarf']

    # CFPD, 23 classes
    if n_cl == 23:
        classes = ['bk', 'T-shirt', 'bag', 'belt', 'blazer', 'blouse', 'coat', 'dress', 'face', 'hair',
                   'hat', 'jeans', 'legging', 'pants', 'scarf', 'shoe', 'shorts', 'skin', 'skirt',
                   'socks', 'stocking', 'sunglass', 'sweater']

    # LIP, 20 classes
    if n_cl == 20:
        classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
                   'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
                   'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
                   'rightShoe']

    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)

    # @evaluation 3: mean IU & per-class IU
    print('IoU for each class:')
    union = num_gt_pix + hist.sum(0) - num_cor_pix

    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IoU', np.nanmean(iu))


if __name__ == "__main__":

    label_path = "gt_0.png"
    pred_path = "pred_0.png"
    n_classes = 18

    # miou
    val_hist = compute_hist(pred_path, label_path, n_classes)
    show_result(val_hist, n_classes)

    print('\n')

    # bfscore
    bfscores, areas_gt = bfscore.bfscore(label_path, pred_path, 2)

    print("\n>>>>BFscore:\n")
    print("BFSCORE:", bfscores)
    print("Per image BFscore:", np.nanmean(bfscores))

    total_area = np.nansum(areas_gt)
    # print("GT area (except background):", total_area)
    fw_bfscore = []
    for each in zip(bfscores, areas_gt):
        if math.isnan(each[0]) or math.isnan(each[1]):
            fw_bfscore.append(math.nan)
        else:
            fw_bfscore.append(each[0] * each[1])
    # print(fw_bfscore)

    print("\n>>>>Weighted BFscore:\n")
    print("Weighted-BFSCORE:", fw_bfscore)
    print("Per image Weighted-BFscore:", np.nansum(fw_bfscore)/total_area)
