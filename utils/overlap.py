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



def get_gaussian(s, sigma=1.0/8):
	temp = np.zeros(s)
	coords = [i // 2 for i in s]
	sigmas = [i * sigma for i in s]
	temp[tuple(coords)] = 1
	gaussian_map = gaussian_filter(temp, sigmas, 0, mode='constant', cval=0)
	gaussian_map /= np.max(gaussian_map)
	return gaussian_map

def add_guass(data):
    img = Image.open('./imgs/test.bmp')
    img = np.array(img)
    patchSize = np.array((256, 256))
    patchStride = patchSize // 2

    result = np.zeros(img.shape)
    normalization = np.zeros(img.shape)
    gaussian_map = get_gaussian(patchSize)
    for i in range(0, img.shape[0] - patchSize[0] + 1, patchStride[0]):
        for j in range(0, img.shape[1] - patchSize[1] + 1, patchStride[1]):
            patch = img[i:i + patchSize[0], j:j + patchSize[1]].astype(np.float32)
            patch *= gaussian_map
            normalization[i:i + patchSize[0], j:j + patchSize[1]] += gaussian_map
            result[i:i + patchSize[0], j:j + patchSize[1]] += patch
    result /= normalization
    plt.imshow(result, cmap=plt.get_cmap('gray'))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def extract_ordered_pathches(img,patch_size,stride_size,pad_way = 'zero'):
    #stride_size 每个图像块的间隔大小
    #path_size 图像块大小
    h,w,c = img.shape
    patch_h,patch_w = patch_size
    stride_h,stride_w = stride_size
    left_h,left_w = (h - patch_h) % stride_h,(w-patch_w) % stride_w
    pad_h ,pad_w = (stride_h - left_h) % stride_h,(stride_w - left_w) %stride_w
    #assert(h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0
    h, w, c = img.shape
   # y方向上的切片数量
    n_pathes_y = (h + pad_h - patch_h) // stride_h + 1
    # x方向上的切片数量
    n_pathes_x = (w + pad_w - patch_w) // stride_w + 1
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
            # print(x1, x2,y1,y2)
            if x2 >= h and y2 >= w:
                return_x1 = x1 - pad_h
                return_x2 = return_x1 + patch_h
                return_y1 = y1 - pad_w
                return_y2 = return_y1 + patch_w
                # print('xy', return_y1, return_y2)
                patches[patch_idx] = img[return_x1:return_x2, return_y1:return_y2]
            elif x2 >= h:
                return_x1 = x1 - pad_h
                return_x2 = return_x1 + patch_h
                # print('x',return_x1,return_x2)
                patches[patch_idx] = img[return_x1:return_x2, y1:y2]
            elif y2 >= w:
                return_y1 = y1 - pad_w
                return_y2 = return_y1 + patch_w
                # print('y',return_y1, return_y2)
                patches[patch_idx] = img[x1:x2, return_y1:return_y2]
            else:
                patches[patch_idx] = img[x1:x2, y1:y2]
            #print(x1, x2, y1, y2)
            patch_idx += 1
    return patches

def rebuild_images(patch_size,stride_size,patches,raw_img_size):

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
    imgs = np.zeros((h , w ))
    #print(imgs.shape,patches[1].shape)
    #图像块之间可能有重叠，所以最后需要除以重复的次数求平均
    weights = np.zeros_like(imgs)

    #重构图像
    # print(n_pathes_x,n_pathes_y)
    patch_idx = 0
    x11 = 0
    for i in range(n_pathes_y):
        y11 = 0
        for j in range(n_pathes_x):
            x1 = i * stride_h
            x2 = x1 + patch_h
            y1 = j * stride_w
            y2 = y1 + patch_w
            # print(x11,x2, y11,y2)
            patch = patches[patch_idx]
            if x2 >= h and y2 >= w:
                return_x1 = x1 - pad_h
                return_x2 = return_x1 + patch_h
                return_y1 = y1 - pad_w
                return_y2 = return_y1 + patch_w
                # print('xy',x11,return_x2,y11,return_y2)
                imgs[x11:return_x2, y11:return_y2] += patch[(patch_h - (return_x2-x11)):patch_h, (patch_w - (return_y2-y11)):patch_w]
            elif x2 >= h:
                return_x1 = x1 - pad_h
                return_x2 = return_x1 + patch_h
                # print('x',x11, return_x2)
                imgs[x11:return_x2, y11:y2] += patch[(patch_h - (return_x2-x11)):patch_h, (patch_w - (y2-y11)):patch_w]
            elif y2 >= w:
                return_y1 = y1 - pad_w
                return_y2 = return_y1 + patch_w
                # print('y',y11, return_y2)
                imgs[x11:x2, y11:return_y2] += patch[(patch_h - (x2-x11)):patch_h, (patch_w - (return_y2-y11)):patch_w]
            else:
                imgs[x11:x2, y11:y2] += patch[(patch_h - (x2-x11)):patch_h, (patch_w - (y2-y11)):patch_w]
            patch_idx += 1
            y11 = y2
        x11 = x2
    return imgs.astype(patches.dtype)



if __name__ == "__main__":
    # main_path = 'D:/Medical Imaging/2Dunet/LungSegData/02Normalized_subsetUR/1GMQX2WE.npy'
    # main_path_ = 'D:/Medical Imaging/2Dunet/LungSegData/01grdtUR/1GMQX2WE.npy'

    main_path = 'D:/Medical Imaging/Synapse data/RawData/Training/img/img0001.nii.gz'
    main_path_ = 'D:/Medical Imaging/Synapse data/RawData/Training/label/label0001.nii.gz'
    image_data = sitk.Image(sitk.ReadImage(main_path))
    mask_data = sitk.Image(sitk.ReadImage(main_path_))
    data_image = sitk.GetArrayFromImage(image_data)
    data_label = sitk.GetArrayFromImage(mask_data)
    data_image = data_image.transpose(1,2,0)
    data_label = data_label.transpose(1,2,0)

    p_size = 256
    patch_size = [p_size, p_size]
    # s_size = round(p_size * (1 / 2))
    s_size = 128
    print(s_size)
    stride_size = [s_size, s_size]
    # Pad_way = 'mirror'
    Pad_way = 'zero'

    # data_image = np.load(main_path)
    # data_label = np.load(main_path_)
    raw_size = data_image.shape

    print(raw_size)
    image = extract_ordered_pathches(data_image,patch_size,stride_size, pad_way=Pad_way)
    label = extract_ordered_pathches(data_label, patch_size,stride_size, pad_way=Pad_way)

    print(image.shape)
    re_image = rebuild_images(patch_size,stride_size, image, raw_img_size = raw_size)
    re_label = rebuild_images(patch_size,stride_size ,label, raw_img_size=raw_size)
    print(np.unique(data_label),np.unique(re_label))
    slice = 75
    plt.figure()
    for i in range(image.shape[0]):
       Size = math.ceil(image.shape[0] ** 0.5)
       plt.subplot(Size, Size, i + 1)
       plt.imshow(image[i, :, :, slice], cmap='gray')
       plt.xticks([])
       plt.yticks([])
    plt.figure()
    for i in range(label.shape[0]):
        Size = math.ceil(label.shape[0] ** 0.5)
        plt.subplot(Size, Size, i + 1)
        plt.imshow(label[i, :, :, slice], cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.figure()
    plt.imshow(re_image[:, :, slice], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.figure()
    plt.imshow(re_label[:, :, slice], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()








