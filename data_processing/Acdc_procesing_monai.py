from configparser import Interpolation
from platform import python_branch
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
    Zoom,
    Resize,
AsChannelLastd,
AsChannelFirstd,
)
import SimpleITK as sitk
import cv2


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
    if x <= crop_size and y >= crop_size:
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
            data_resize = cv2.resize(pan_img[:,:,i],(crop_size,crop_size),interpolation = cv2.INTER_NEAREST)
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
            data_resize = cv2.resize(pan_img[:,:,i],(crop_size,crop_size),interpolation = cv2.INTER_NEAREST)
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
            data_resize = cv2.resize(img[:,:,i],(crop_size,crop_size),interpolation = cv2.INTER_NEAREST)
            data_list.append(data_resize)
        crop_img = np.stack(data_list)
        crop_img = crop_img.transpose(1, 2, 0)

    return crop_img
    

def resize_img(img, img_size):
    x, y, z = img.shape
    data_list = []
    for i in range(z):
        data_resize = cv2.resize(img[:,:,i],(img_size,img_size),interpolation = cv2.INTER_NEAREST)
        data_list.append(data_resize)
    re_img = np.stack(data_list)
    re_img = re_img.transpose(1, 2, 0)
    # print(re_img.shape)
    return re_img


def pad_resize(img, img_size):
    x,y,z = img.shape
    if x>y:
        pad_left = (x-y)//2+(x-y)%2
        pad_right = (x-y)//2
        image = np.pad(img,((0,0),(pad_left,pad_right),(0,0)),'constant')
    elif y>x:
        pad_up = (y-x)//2+(y-x)%2
        pad_down = (y-x)//2
        image = np.pad(img,((pad_up,pad_down),(0,0),(0,0)),'constant')
    else:
        image = img
    x,y,z = image.shape
    print(x,y,z)
    data_list = []
    for i in range(z):
        data_resize = cv2.resize(image[:,:,i],(img_size,img_size),interpolation = cv2.INTER_NEAREST)
        data_list.append(data_resize)
    re_img = np.stack(data_list)
    re_img = re_img.transpose(1, 2, 0)
    return re_img

def pad_resize(img, img_size):
    x,y,z = img.shape
    if x>y:
        pad_left = (x-y)//2+(x-y)%2
        pad_right = (x-y)//2
        image = np.pad(img,((0,0),(pad_left,pad_right),(0,0)),'constant')
    elif y>x:
        pad_up = (y-x)//2+(y-x)%2
        pad_down = (y-x)//2
        image = np.pad(img,((pad_up,pad_down),(0,0),(0,0)),'constant')
    else:
        image = img
    
    return image

def resampleVolume(data_dict,new_spacing):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    # print(data_dict)

    print('raw shape',data_dict['image'].shape,data_dict['label'].shape)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 重采样
    spacing = Spacingd(keys=["image", "label"], pixdim=(1.56, 1.56, new_spacing), mode=("bilinear", "nearest"))
    data_dict = spacing(data_dict)

    data_image, data_label = data_dict["image"], data_dict["label"]
    # print(data_label.shape)
    # # z = data_label.shape[-1]
    # # print(total_slice)
    # total_slice = total_slice + data_label.shape[-1]
    return data_image, data_label

def preprossing_data(data_dicts,save_path,patient_name,new_spacing,crop_size):
    save_image = save_path
    save_label = save_path.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    image,label = resampleVolume(data_dicts,new_spacing)
    
    image = np.squeeze(image)
    label = np.squeeze(label)
    print('re shape',image.shape,label.shape)

    image -= image.mean()
    image /= image.std()

    # image = (image - image.min()) / (image.max()-image.min())      #ImageTensor = ImageTensor/max(ImageTensor(:));
    # nor_image[nor_image<0] = 0
    # nor_image[nor_image>1] = 1

    # crop_size = 256

    image = crop(image, crop_size)
    label = crop(label, crop_size)

    print('crop shape',image.shape,label.shape)

    print('finally shape',image.shape,label.shape)
    print('finally unique',np.unique(image),np.unique(label))

    label_raw = [0.,1.,2.,3.]
    label_fy = list(np.unique(label))
    # print(label_raw,label_fy)
    assert label_raw == label_fy,'label type error!'
    
    # # '''
    show = False
    if show:
        x,y,z = image.shape
        plt.figure()
        show_num = z
        sh = int(z ** 0.5)
        for i in range(0,show_num):
            num = i
            x = image[:,:,num]
            plt.subplot(sh,sh+3,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,show_num):
            num = i
            x = label[:,:,num]
            plt.subplot(sh,sh+3,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()
    np.save(save_image + patient_name + '.npy', image)
    np.save(save_label + patient_name + '.npy', label)
    print('\n')

if __name__ == '__main__':
    crop_size = 256
    x = 260
    y = 200
    z = 10
    img = np.zeros((x,y,z))
    center_index_x = x // 2  # 中心坐标
    center_index_y = y // 2  # 中心坐标
    crop_size_num = crop_size // 2
    crop_img = img[center_index_x - crop_size_num:center_index_x + crop_size_num,
               center_index_y - crop_size_num:center_index_y + crop_size_num, :]
    data_dicts = {'image': img}
    chanal = AddChanneld(keys=["image"])
    img_ = chanal(data_dicts)
    print(img_['image'].shape)
    re = Resize(spatial_size = (x,crop_size,z))
    z = re(img_['image'])
    print(img.shape,crop_img.shape,z.shape)



    '''
    python Acdc_procesing_monai.py --crop_size 256 --Spacing_z 3
    '''
#     import re
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--Spacing_z', type=str, default=0)
#     parser.add_argument('--crop_size', type=int, default=0)
#     args = parser.parse_args()
#
#     # Spacing = int(re.findall(r"\d+\.?\d*",args.Spacing_z)[0])
#     print('sapcing',args.Spacing_z)
#     file_name = 'acdc_100.txt'
#     main_path = '/media/xd/date/muzhaoshan/ACDC/'
#     file_list = open(os.path.join(main_path, file_name)).readlines()
#     total_slice = 0
#     new_spacing = 3
#     for i in range(len(file_list)):
#         patient_num = file_list[i].strip('\n')
#         patient_name = patient_num
#         print(patient_name)
#         patient_path = main_path + 'raw_data/patient' + patient_num + '/'
#         patient_file_name = sorted(os.listdir(patient_path))
#
#         img_ED_path = patient_path + patient_file_name[2]
#         gt_ED_path = patient_path + patient_file_name[3]
#         data_dicts_ED = {'image': img_ED_path, 'label': gt_ED_path}
#
#         # raw_data = sitk.Image(sitk.ReadImage(img_ED_path))
#         # raw_spacing = raw_data.GetSpacing()
#         save_path_ED = main_path + 're_nor_ED_'+args.Spacing_z+'/images/'
#         preprossing_data(data_dicts_ED, save_path_ED,new_spacing,args.crop_size)
#         # total_slice = resampleVolume(data_dicts_ED,total_slice)
#         # print(total_slice)
#
#
#         img_ES_path = patient_path + patient_file_name[4]
#         gt_ES_path = patient_path + patient_file_name[5]
#         data_dicts_ES = {'image': img_ES_path, 'label': gt_ES_path}
#
#         # raw_data = sitk.Image(sitk.ReadImage(img_ES_path))
#         # raw_spacing = raw_data.GetSpacing()
#         save_path_ES = main_path + 're_nor_ES_'+args.Spacing_z+'/images/'
#         preprossing_data(data_dicts_ES, save_path_ES,new_spacing,args.crop_size)
#         # total_slice = resampleVolume(data_dicts_ES,total_slice)
# #         print(total_slice)
# # print(total_slice)
#
#
#
#
#
#
#
#
