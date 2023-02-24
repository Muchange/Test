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

    resample_image = resampleVolume([1.0,1.0,3.0],image_data,is_label = False)
    resample_mask = resampleVolume([1.0,1.0,3.0],mask_data,is_label = True)

    image = sitk.GetArrayFromImage(resample_image)
    label = sitk.GetArrayFromImage(resample_mask)

    raw_label_list = [0,8,4,3,2,6,11,1,7]
    replace_label_list = [0,88,44,33,22,66,111,112,77]
    new_label_list = [0,1,2,3,4,5,6,7,8]
    # print('raw label',np.unique(label))
    for i in range(len(raw_label_list)):
        label[label == raw_label_list[i]] = replace_label_list[i]
    # print('raw label',np.unique(label))
    for i in range(len(raw_label_list)):
        label[label < 20] = 0
    # print('replace label',np.unique(label))
    for i in range(len(raw_label_list)):
        label[label == replace_label_list[i]] = new_label_list[i]
    # print('new label',np.unique(label))
    
    #normalized
    MAX = 275.0
    MIN = -175.0
    nor_image = (image - MIN) / (MAX-MIN)      #min=-1024 %ImageTensor = ImageTensor/max(ImageTensor(:));
    nor_image[nor_image<0] = 0
    nor_image[nor_image>1] = 1
    # print('unique',np.unique(nor_image),np.unique(label))
    nor_image = nor_image.transpose(1,2,0)
    nor_image =np.flipud(nor_image)
    nor_image =np.fliplr(nor_image)
    label = label.transpose(1,2,0)
    label = np.flipud(label)
    label = np.fliplr(label)
    print('finally shape',nor_image.shape,label.shape)
    print('finally unique')
    print(np.unique(nor_image),np.unique(label))
    
    # '''
    show = False
    if show:
        index = 0
        for i in range(label.shape[-1]):
            if label[:,:,i].max()>1:
                index = i
                break;
        plt.figure()
        slice_index = index
        # print(nor_image[:,:,slice_index])
        print(slice_index)
        show_num = 4
        sub_plt = int(pow(show_num,0.5))
        for i in range(0,show_num):
            num = i+slice_index
            x = nor_image[:,:,num]
            plt.subplot(sub_plt,sub_plt,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,show_num):
            num = i+slice_index
            x = label[:,:,num]
            plt.subplot(sub_plt,sub_plt,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()
    print('\n')
    # '''
    np.save(save_image + patient_name+'.npy',nor_image)
    np.save(save_label + patient_name+'.npy',label)
    # return nor_image,label
if __name__ == '__main__':
    file_list = open(os.path.join('D:/Medical Imaging/Synapse data/', 'Sy_test.txt')).readlines()
    main_path = 'D:/Medical Imaging/Synapse data/RawData/'
    input_path = main_path + '/Training/img/'
    save_path = main_path + 're_nor/images/'
    for i in range(1):
        patient_name = file_list[i].strip('\n')
        print(patient_name)
        data_image_str = input_path +'img' + patient_name + '.nii.gz'
        data_masks_str = input_path.replace('img', 'label') + 'label' + patient_name + '.nii.gz'
        preprossing_data(data_image_str,data_masks_str,patient_name,save_path)


