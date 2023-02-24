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
    CropForegroundd,
    Orientationd
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

def data_procesing(data_in):
    data_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-125, a_max=275,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    data_ds = data_transform(data_in)
    print(data_ds["image"].shape,data_ds["label"].shape)

    # np.save('D:/Medical Imaging/BraTS2021 Task1/BraTS2021_Training_Data/test.npy',image)


if __name__ == '__main__':
    # main_path = 'D:/Medical Imaging/BraTS2021 Task1/BraTS2021_Training_Data/'
    #
    # crop_size = [96,96,96]
    #
    # train,val = datafold_read(datalist=main_path + 'brats21_test.json', basedir=main_path, fold=1)
    # print(train,val)
    # data_procesing(data_in = train,data_type = 'train',in_path = main_path,crop_size = crop_size)
    main_path = 'D:/Medical Imaging/Synapse data/'
    file_list = open(os.path.join(main_path, 'Sy_train.txt')).readlines()

    input_path = main_path + 'RawData/Training/img/'
    save_path = main_path + 'RawData/Training/label/'
    for i in range(len(file_list)):
        patient_name = file_list[i].strip('\n')
        print(patient_name)
        data_image_str = input_path + 'img' + patient_name + '.nii.gz'
        data_masks_str = input_path.replace('img', 'label') + 'label' + patient_name + '.nii.gz'
        data_dict = {'image': data_image_str, 'label': data_masks_str}
        data_procesing(data_dict)




