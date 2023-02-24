from configparser import Interpolation
from re import A
import re
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import torch
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    RandFlipd,
    RandRotate90d,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandAffined,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandCropByPosNegLabeld,
ConvertToMultiChannelBasedOnBratsClassesd,CropForegroundd,
)
import SimpleITK as sitk
import cv2
from monai import data, transforms
import json

def datafold_read(datalist, basedir, fold=1, key="training"):

    with open(datalist) as f:
        json_data = json.load(f)
    # print(json_data)
    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    # print(tr,val)
    return tr, val

def data_procesing(data_in,data_type,in_path,crop_size):
    if data_type =='train':
        data_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.CropForegroundd(
                    keys=["image", "label"], source_key="image", k_divisible=crop_size
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                # transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif data_type == 'val' or data_type == 'test':
        data_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                # transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        assert data_type == 'error','data_type error!'

    data_ds = data.Dataset(data=data_in, transform=data_transform)
    for idx, batch_data in enumerate(data_ds):
        print('---------------------------',idx)
        image, label = batch_data["image"], batch_data["label"]
        print(image.shape, label.shape)
        np.save('D:/Medical Imaging/BraTS2021 Task1/BraTS2021_Training_Data/test.npy',image)


if __name__ == '__main__':
    main_path = 'D:/Medical Imaging/BraTS2021 Task1/BraTS2021_Training_Data/'

    crop_size = [96,96,96]

    train,val = datafold_read(datalist=main_path + 'brats21_test.json', basedir=main_path, fold=1)
    # print(train,val)
    # data_procesing(data_in = train,data_type = 'train',in_path = main_path,crop_size = crop_size)









