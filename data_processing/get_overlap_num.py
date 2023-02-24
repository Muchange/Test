from tkinter import image_names
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk
import torch


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list


def extract_ordered_pathches(img,stride_size,patch_size,pad_way = 'zero'):
    #stride_size 每个图像块的间隔大小
    #path_size 图像块大小
    h,w,c = img.shape
    patch_h,patch_w = patch_size
    stride_h,stride_w = stride_size
    left_h,left_w = (h - patch_h) % stride_h,(w-patch_w) % stride_w
    pad_h ,pad_w = (stride_h - left_h) % stride_h,(stride_w - left_w) %stride_w
    # print(pad_h ,pad_w)
    if pad_h:
        # 设置填充后的图像大小
        pad_imgs = np.zeros((h + pad_h, w, c), dtype=img.dtype)
        #这部分位置用来放置原图
        start_y = pad_h //2
        end_y = start_y + h
        #中间部分设计
        pad_imgs[start_y:end_y,:,:] = img
        if pad_way=='mirror':
            #上下填充(镜像)
            pad_imgs[:start_y, :, :] = img[:start_y,:,:][::-1]
            pad_imgs[end_y:, :, :] = img[h-(pad_h - pad_h//2):, :, :][::-1]
        elif pad_way == 'zero':
            #上下填充(0)
            pad_imgs[:start_y, :, :] = 0
            pad_imgs[end_y:, :, :] = 0
        img = pad_imgs
    if pad_w:
        h = img.shape[0]
        #print(img.shape,pad_imgs.shape)
        # 设置填充后的图像大小
        pad_imgs = np.zeros((h, w + pad_w, c), dtype=img.dtype)
        #print(img.shape, pad_imgs.shape)
        # 这部分位置用来放置原图
        start_x = pad_w // 2
        end_x = start_x + w
        # 中间部分设计
        pad_imgs[ :, start_x:end_x,:] = img
        if pad_way=='mirror':
            #上下填充(镜像)
            pad_imgs[:, :start_x, :] = img[:, :start_x, :][:,::-1]
            pad_imgs[:, end_x:, :] = img[:, w - (pad_w - pad_w // 2):, :][:,::-1]
        elif pad_way == 'zero':
            # 上下填充(0)
            pad_imgs[:, :start_y, :] = 0
            pad_imgs[:, end_y:, :] = 0
        img = pad_imgs

    #assert(h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0
    h, w, c = img.shape
    #y方向上的切片数量
    n_pathes_y = (h - patch_h) // stride_h + 1
    # x方向上的切片数量
    n_pathes_x = (w - patch_w) // stride_w + 1
    #每张图片的切片数量
    n_pathches_per_img = n_pathes_y * n_pathes_x
    #print(h,patch_h,stride_h,w,patch_w,stride_w,n_pathes_y,n_pathes_x)
    #切片总数
    n_pathches = n_pathches_per_img
    #设置图像块大小
    patches = np.zeros((n_pathches,patch_h,patch_w,c),dtype=img.dtype)
    patch_idx = 0

    #依次对每张图片处理
    #从左到右，从上到下
    for i in range(n_pathes_y):
        for j in range(n_pathes_x):
            x1 = i * stride_h
            x2 = x1 + patch_h
            y1 = j * stride_w
            y2 = y1 + patch_w
            #print(x1, x2, y1, y2)
            patches[patch_idx] = img[x1:x2, y1:y2,:]
            patch_idx += 1
    return patches

def rebuild_images(stride_size,patch_size,patches,raw_img_size):

    # stride_size 每个图像块的间隔大小
    # path_size 图像块大小
    h, w = raw_img_size
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride_size
    left_h, left_w = (h - patch_h) % stride_h, (w - patch_w) % stride_w
    pad_h, pad_w = (stride_h - left_h) % stride_h, (stride_w - left_w) % stride_w
    # y方向上的切片数量
    n_pathes_y = (h + pad_h - patch_h) // stride_h + 1
    # x方向上的切片数量
    n_pathes_x = (w + pad_w - patch_w) // stride_w + 1
    # 每张图片的切片数量
    n_pathches_per_img = n_pathes_y * n_pathes_x
    # 切片总数
    #n_pathches = n_patches // n_pathches_per_img
    # 设置图像块大小
    imgs = np.zeros((h + pad_h, w + pad_w))
    #print(imgs.shape,patches[1].shape)
    #图像块之间可能有重叠，所以最后需要除以重复的次数求平均
    weights = np.ones_like(imgs)

    # #重构图像
    patch_idx = 0
    for i in range(n_pathes_y):
        for j in range(n_pathes_x):
            x1 = i * stride_h
            x2 = x1 + patch_h
            y1 = j * stride_w
            y2 = y1 + patch_w
            #print(x1,x2, y1,y2)
            patch_idx = n_pathes_x * i + j
            # print(patches[patch_idx].shape)
            imgs[x1:x2, y1:y2] += patches[patch_idx]
            weights[x1:x2, y1:y2] += 1
    #重叠部分求平均
    imgs = imgs/weights
    start_y = pad_h //2
    end_y = start_y + h
    start_x = pad_w // 2
    end_x = start_x + w
    imgs = imgs[start_y:end_y,start_x:end_x]
    return imgs.astype(patches.dtype)


if __name__ == "__main__":
    image_size=  256
    patch_size = [image_size,image_size]
    stride_size = [224,224] 
    main_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/re_nor/images/'
    file_list = os.listdir(main_path)
    for i in range(len(file_list)):
        name = file_list[i]
        data = np.load(main_path + name)
        image = extract_ordered_pathches(data, stride_size, patch_size, pad_way='zero')
        # if image.shape[0] <5 :
        print(data.shape[0],image.shape[0])
        # print(image.shape)

