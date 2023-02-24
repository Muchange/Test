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
)
import SimpleITK as sitk
import cv2


def resampleVolume(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    # print(data_dict)

    # print(data_dict['image'].shape,data_dict['label'].shape)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 重采样
    spacing = Spacingd(keys=["image", "label"], pixdim=(1.56, 1.56, 5.0), mode=("bilinear", "nearest"))
    data_dict = spacing(data_dict)

    data_image, data_label = data_dict["image"], data_dict["label"]

    return data_image, data_label

def crop(img, crop_size):
    x, y, z = img.shape
    # print(x,y,z)
    if x <= crop_size and y > crop_size:
        pad_x = (crop_size - x)
        # print(pad_x)
        # 设置填充后的图像大小
        pad_imgs = np.empty((x + pad_x, y, z), dtype=img.dtype)
        # print(pad_imgs.shape)
        # 这部分位置用来放置原图
        start_x = pad_x // 2
        end_x = start_x + x


        # 中间部分设计
        # print(start_x,end_x,start_y,end_y)
        pad_imgs[start_x:end_x,:, :] = img
        # 上下填充
        pad_imgs[:start_x, :, :] = 0
        pad_imgs[end_x:, :, :] = 0
        pan_img = pad_imgs

        x,y,z = pan_img.shape
        center_index_x = x // 2  # 中心坐标
        center_index_y = y // 2  # 中心坐标
        crop_size_num = crop_size // 2
        crop_img = pan_img[center_index_x - crop_size_num:center_index_x + crop_size_num,
                   center_index_y - crop_size_num:center_index_y + crop_size_num, :]

    elif y <= crop_size and x >= crop_size:
        pad_y = (crop_size - y)
        # print(pad_y)
        # 设置填充后的图像大小
        pad_imgs = np.empty((x, y + pad_y, z), dtype=img.dtype)
        # print(pad_imgs.shape)
        # 这部分位置用来放置原图
        start_y = pad_y // 2
        end_y = start_y + y

        # 中间部分设计
        # print(start_x,end_x,start_y,end_y)
        pad_imgs[:, start_y:end_y, :] = img
        # 左右填充
        pad_imgs[:, :start_y, :] = 0
        pad_imgs[:, end_y:, :] = 0
        pan_img = pad_imgs

        x, y, z = pan_img.shape
        center_index_x = x // 2  # 中心坐标
        center_index_y = y // 2  # 中心坐标
        crop_size_num = crop_size // 2
        crop_img = pan_img[center_index_x - crop_size_num:center_index_x + crop_size_num,
                   center_index_y - crop_size_num:center_index_y + crop_size_num, :]
        # print(crop_img.shape)
    elif x <= crop_size and y <= crop_size:
        pad_x = (crop_size - x)
        pad_y = (crop_size - y)
        # 设置填充后的图像大小
        pad_imgs = np.empty((x + pad_x, y + pad_y, z), dtype=img.dtype)

        # 这部分位置用来放置原图
        start_x = pad_x // 2
        end_x = start_x + x

        start_y = pad_y // 2
        end_y = start_y + y

        # 中间部分设计
        pad_imgs[start_x:end_x, start_y:end_y, :] = img
        # 上下填充
        pad_imgs[:start_x, :, :] = 0
        pad_imgs[end_x:, :, :] = 0
        # 左右填充
        pad_imgs[:, :start_y, :] = 0
        pad_imgs[:, end_y:, :] = 0
        crop_img = pad_imgs
    else:
        center_index_x = x // 2  # 中心坐标
        center_index_y = y // 2  # 中心坐标
        crop_size_num = crop_size // 2
        crop_img = img[center_index_x - crop_size_num:center_index_x + crop_size_num,
                   center_index_y - crop_size_num:center_index_y + crop_size_num, :]
    # print(crop_img.shape)
    return crop_img


def crop_resize(img, crop_size):
    x, y, z = img.shape
    # print(x,y,z)
    if x <= crop_size and y > crop_size:
        pad_x = (crop_size - x)
        # print(pad_x)
        # 设置填充后的图像大小
        pad_imgs = np.empty((x + pad_x, y, z), dtype=img.dtype)
        # print(pad_imgs.shape)
        # 这部分位置用来放置原图
        start_x = pad_x // 2
        end_x = start_x + x

        # 中间部分设计
        # print(start_x,end_x,start_y,end_y)
        pad_imgs[start_x:end_x,:, :] = img
        # 上下填充
        pad_imgs[:start_x, :, :] = 0
        pad_imgs[end_x:, :, :] = 0
        pan_img = pad_imgs

        x,y,z = pan_img.shape
        data_list = []
        for i in range(z):
            data_resize = cv2.resize(pan_img[:,:,i],(crop_size,crop_size),interpolation = cv2.INTER_CUBIC)
            data_list.append(data_resize)
        crop_img = np.stack(data_list)
        crop_img = crop_img.transpose(1, 2, 0)

    elif y <= crop_size and x >= crop_size:
        pad_y = (crop_size - y)
        # print(pad_y)
        # 设置填充后的图像大小
        pad_imgs = np.empty((x, y + pad_y, z), dtype=img.dtype)
        # print(pad_imgs.shape)
        # 这部分位置用来放置原图
        start_y = pad_y // 2
        end_y = start_y + y

        # 中间部分设计
        # print(start_x,end_x,start_y,end_y)
        pad_imgs[:, start_y:end_y, :] = img
        # 左右填充
        pad_imgs[:, :start_y, :] = 0
        pad_imgs[:, end_y:, :] = 0
        pan_img = pad_imgs

        x, y, z = pan_img.shape
        data_list = []
        for i in range(z):
            data_resize = cv2.resize(pan_img[:,:,i],(crop_size,crop_size),interpolation = cv2.INTER_CUBIC)
            data_list.append(data_resize)
        crop_img = np.stack(data_list)
        crop_img = crop_img.transpose(1, 2, 0)

    elif x <= crop_size and y <= crop_size:
        pad_x = (crop_size - x)
        pad_y = (crop_size - y)
        # 设置填充后的图像大小
        pad_imgs = np.empty((x + pad_x, y + pad_y, z), dtype=img.dtype)

        # 这部分位置用来放置原图
        start_x = pad_x // 2
        end_x = start_x + x

        start_y = pad_y // 2
        end_y = start_y + y

        # 中间部分设计
        pad_imgs[start_x:end_x, start_y:end_y, :] = img
        # 上下填充
        pad_imgs[:start_x, :, :] = 0
        pad_imgs[end_x:, :, :] = 0
        # 左右填充
        pad_imgs[:, :start_y, :] = 0
        pad_imgs[:, end_y:, :] = 0
        crop_img = pad_imgs
    else:
        x,y,z = img.shape
        # print(x,y,z)
        data_list = []
        for i in range(z):
            data_resize = cv2.resize(img[:,:,i],(crop_size,crop_size),interpolation = cv2.INTER_CUBIC)
            data_list.append(data_resize)
        crop_img = np.stack(data_list)
        crop_img = crop_img.transpose(1, 2, 0)

    return crop_img
    

def resize_img(img, img_size):
    x, y, z = img.shape
    data_list = []
    for i in range(z):
        data_resize = cv2.resize(img[:,:,i],(img_size,img_size),interpolation = cv2.INTER_CUBIC)
        data_list.append(data_resize)
    re_img = np.stack(data_list)
    re_img = re_img.transpose(1, 2, 0)
    # print(re_img.shape)
    return re_img

def preprossing_data(path_img,path_label,patient_name,save_path):
    save_image = save_path
    save_label = save_path.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    image_data = sitk.Image(sitk.ReadImage(path_img))
    mask_data = sitk.Image(sitk.ReadImage(path_label))

    image = sitk.GetArrayFromImage(image_data)
    label = sitk.GetArrayFromImage(mask_data)

    image = image.transpose(1, 2, 0)
    label = label.transpose(1, 2, 0)

    print(image.shape,label.shape)

    np.save(save_image + patient_name + '.npy', image)
    np.save(save_label + patient_name + '.npy', label)

if __name__ == '__main__':
    file_name = 'train.txt'
    main_path = 'D:/Medical Imaging/ACDC/ACDC_data/'
    file_list = open(os.path.join(main_path, file_name)).readlines()
    for i in range(len(file_list)):
        patient_num = file_list[i].strip('\n')
        patient_name = patient_num
        print(patient_name)
        patient_path = main_path + '/patient' + patient_num + '/'
        patient_file_name = sorted(os.listdir(patient_path))

        img_ED_path = patient_path + patient_file_name[2]
        gt_ED_path = patient_path + patient_file_name[3]
        img_ES_path = patient_path + patient_file_name[4]
        gt_ES_path = patient_path + patient_file_name[5]

        save_path_ED = 'D:/Medical Imaging/ACDC/raw_data/images/'
        preprossing_data(img_ED_path, gt_ED_path, patient_name, save_path_ED)
        save_path_ES =  'D:/Medical Imaging/ACDC/raw_data/images/'
        preprossing_data(img_ES_path, gt_ES_path, patient_name, save_path_ES)






