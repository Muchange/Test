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
)
import SimpleITK as sitk
import cv2 as cv

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

def resampleVolume(re_spacing, raw_data,is_label = False):
    """
    将体数据重采样的指定的spacing大小
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    raw_data：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]

    # 读取文件的size和spacing信息
    raw_size = raw_data.GetSize()
    raw_spacing = raw_data.GetSpacing()

    # print('raw shape',raw_size)
    # print('raw spacing', raw_spacing)

    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(raw_size[0] * (raw_spacing[0] / re_spacing[0]))
    outsize[1] = int(raw_size[1] * (raw_spacing[1] / re_spacing[1]))
    outsize[2] = int(raw_size[2] * (raw_spacing[2] / re_spacing[2]))

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(raw_data)
    resampler.SetSize(outsize)
    resampler.SetOutputOrigin(raw_data.GetOrigin())
    resampler.SetOutputDirection(raw_data.GetDirection())
    resampler.SetOutputSpacing(re_spacing)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        #resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkFloat64)  # 线性插值用于PET/CT/MRI之类的，保存float32

    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    re_spacing_img = resampler.Execute(raw_data)  # 得到重新采样后的图像
    return re_spacing_img


def preprossing_data(path_img,path_label,patient_name,save_path):

    save_image = save_path
    save_label = save_path.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    image_data = sitk.Image(sitk.ReadImage(path_img))
    mask_data = sitk.Image(sitk.ReadImage(path_label)) 

    resample_image = resampleVolume([1.5,1.5,10.0],image_data,is_label = False)
    resample_mask = resampleVolume([1.5,1.5,10.0],mask_data,is_label = True)

    image = sitk.GetArrayFromImage(resample_image)
    label = sitk.GetArrayFromImage(resample_mask)

    print('raw shape',image.shape,label.shape)

    image = image.transpose(1, 2, 0)
    label = label.transpose(1, 2, 0)
    # print('finally shape', image.shape, label.shape)

    image -= image.mean()
    image /= image.std()
    crop_size = 256

    image = crop(image, crop_size)
    label = crop(label, crop_size)

    print('finally shape',image.shape,label.shape)
    # print('finally unique')
    # print(np.unique(image),np.unique(label))
    
    # # '''
    show = False
    if show:
        x,y,z = image.shape
        plt.figure()
        show_num = z
        for i in range(0,show_num):
            num = i
            x = image[:,:,num]
            plt.subplot(3,4,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,show_num):
            num = i
            x = label[:,:,num]
            plt.subplot(3,4,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()
    print('\n')
    # # '''
    np.save(save_image + patient_name+'.npy',image)
    np.save(save_label + patient_name+'.npy',label)
    # return nor_image,label
if __name__ == '__main__':
    file_name = 'train.txt'
    main_path = 'D:/Medical Imaging/ADAC/ACAD_data/'
    file_list = open(os.path.join(main_path, file_name)).readlines()
    for i in range(len(file_list)):
        patient_num = file_list[i].strip('\n')
        patient_name = patient_num
        print(patient_name)
        patient_path = main_path + 'patient' + patient_num + '/'
        patient_file_name = sorted(os.listdir(patient_path))

        img_ED_path = patient_path + patient_file_name[2]
        gt_ED_path = patient_path + patient_file_name[3]
        img_ES_path = patient_path + patient_file_name[4]
        gt_ES_path = patient_path + patient_file_name[5]

        save_path_ED = main_path + 're_nor_ED/images/'
        preprossing_data(img_ED_path,gt_ED_path,patient_name,save_path_ED)
        save_path_ES = main_path + 're_nor_ES/images/'
        preprossing_data(img_ES_path,gt_ES_path,patient_name,save_path_ES)


