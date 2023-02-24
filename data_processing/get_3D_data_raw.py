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
# from resample_normalized_Synapse  import *
import SimpleITK as sitk

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


def crop(img, crop_size):
    x, y, z = img.shape
    pad = (crop_size - x)
    if x < crop_size or y < crop_size:
        # 设置填充后的图像大小
        pad_imgs = np.empty((x + pad, y + pad, z), dtype=img.dtype)

        # 这部分位置用来放置原图
        start_x = pad // 2
        end_x = start_x + x

        start_y = pad // 2
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
        center_index = x // 2  # 中心坐标
        crop_size_num = crop_size // 2
        crop_img = img[center_index - crop_size_num:center_index + crop_size_num,
                   center_index - crop_size_num:center_index + crop_size_num, :]
    return crop_img


def pandding_3D(image, label, patch_size, pandding):
    w, h, z = image.shape
    pad_imgs = np.empty((w, h, z + pandding * 2), dtype=image.dtype)
    pad_label = np.empty((w, h, z + pandding * 2), dtype=label.dtype)
    print(pad_imgs.shape)
    start_z = pandding
    end_z = z + pandding

    pad_imgs[:, :, start_z:end_z] = image
    pad_label[:, :, start_z:end_z] = label

    pad_imgs[:, :, :start_z] = image[:, :, :start_z]
    pad_imgs[:, :, end_z + patch_size:] = image[:, :, end_z:]

    pad_label[:, :, :start_z] = label[:, :, :start_z]
    pad_label[:, :, end_z + patch_size:] = label[:, :, end_z:]

    return pad_imgs, pad_label



def Augm_flip(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 随机翻转，xy面
    randfilp = RandFlipd(keys=["image", "label"], prob=1, spatial_axis=(0, 1))
    data_dict = randfilp(data_dict)


    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_rotate(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 随机90度旋转
    randrotate90 = RandRotate90d(keys=["image", "label"], prob=1, spatial_axes=(0, 1))
    data_dict = randrotate90(data_dict)

    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_Rand3DElasticd(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    image = data_dict["image"]
    x, y, z = image.shape

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 弹性变换
    randaelasticd = Rand3DElasticd(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(300, 500),
        spatial_size=(x, y, z),
        translate_range=(14, 14, 0),
        rotate_range=(np.pi / 36, np.pi / 36, 0),
        scale_range=(0.15, 0.15, 0),
        padding_mode="border",
    )

    data_dict = randaelasticd(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_RandGaussianNoise(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 高斯噪声
    randgaussiannoise = RandGaussianNoised(keys=["image", "label"],prob=1, mean=0.0, std=0.05)

    data_dict = randgaussiannoise(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label


def Augm_RandGaussianSharpen(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    image = data_dict["image"]

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 高斯锐化（使用高斯模糊）
    randgaussiansharpen = RandGaussianSharpend(
                        keys=["image", "label"],
                        prob=1,
                        sigma1_x=(0.5, 1.0),
                        sigma1_y=(0.5, 1.0),
                        sigma1_z=(0.5, 1.0),
                        sigma2_x=0.5,
                        sigma2_y=0.5,
                        sigma2_z=0.5,
                        alpha=(10.0, 30.0),
                        approx='erf')

    data_dict = randgaussiansharpen(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label

def Augm_RandAffined(data_dict):
    # 读取
    loader = LoadImaged(keys=["image", "label"])
    data_dict = loader(data_dict)
    image = data_dict["image"]
    x, y, z = image.shape

    # 添加通道
    add_channel = AddChanneld(keys=["image", "label"])
    data_dict = add_channel(data_dict)

    # 弹性变换
    randaffined = RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spatial_size=(x, y, z),
        rotate_range=(np.pi / 36, np.pi / 36, 0),
        scale_range=(0.15, 0.15, 0),
        padding_mode="border",
    )

    data_dict = randaffined(data_dict)
    data_image, data_label = data_dict["image"], data_dict["label"]
    data_image = data_image.squeeze(0)
    data_label = data_label.squeeze(0)
    return data_image, data_label

def get_3D_patch_slice(data_image, data_masks, patch_size, crop_size,
                             save_patch_name, tumor_slice_name, output_path, patient_name):
    save_image = output_path
    save_label = output_path.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    image = crop(data_image, crop_size)
    label = crop(data_masks, crop_size)

    print(image.shape)
    # plt.figure()
    x, y, z = image.shape
    pandding = int((patch_size - 1) / 2)
    pad_img, pad_label = pandding_3D(image, label, patch_size, pandding)
    #####save include tumor slice patch
    no_tumor = []  ##保存包含tumor slice的索引
    for i in range(z):
        if label[:, :, i].max() == 1:
            no_tumor.append(i)
            index = pandding + i
            img = pad_img[:, :, index - pandding:index + pandding + 1]
            mask = pad_label[:, :, index - pandding:index + pandding + 1].astype(np.int64)
            np.save(save_image + str(save_patch_name) + '.npy', img)
            np.save(save_label + str(save_patch_name) + '.npy', mask)
            print(patient_name, '  include tumor slice:', i,
                  '  save slice number name:', tumor_slice_name,
                  '  total slice :', z, '  save name patch:', save_patch_name)
            save_patch_name += 1
            tumor_slice_name += 1
            # print(name_index)

    # 在不包含tumor数据中，只保存训练集，验证集不保存
    if len(no_tumor) <= 0:
        return save_patch_name, tumor_slice_name
    else:
        print(no_tumor, no_tumor[0], no_tumor[-1], len(no_tumor))
        #####save no tumor slice patch
        ##取不包含tumor slice的个数，数量为包含tumor的二分之一
        ##从0到在0到包含第一张tumor的slice范围中取包含slice的一半
        ##从包含最后张tumor的slice到整个病人最后一张slice范围中取包含slice的一半
        no_tumor_number = int(len(no_tumor) * (3/4) )  ##取不包含tumor数量（4倍）
        if no_tumor[0] <= no_tumor_number:
            no_tumor_number = no_tumor[0]
        else:
            no_tumor_number = no_tumor_number
        first_slice = sorted(random.sample(range(0, no_tumor[0]), no_tumor_number))  # 随机在0到包含第一张tumor的slice范围中取
        if (z - no_tumor[-1]-1) <= no_tumor_number:
            no_tumor_number = (z - no_tumor[-1]) - 1
        else:
            no_tumor_number = no_tumor_number
        finally_slice = sorted(random.sample(range(no_tumor[-1] + 1, z), no_tumor_number))  # 随机在0到包含第一张tumor的slice范围中取
        print(first_slice, finally_slice, no_tumor_number)
        ##保存没有tumor的前面部分
        for i_f in range(no_tumor_number):
            index_first = pandding + first_slice[i_f]
            index = index_first
            img = pad_img[:, :, index - pandding:index + pandding + 1]
            mask = pad_label[:, :, index - pandding:index + pandding + 1].astype(np.int64)

            np.save(save_image + str(save_patch_name) + '.npy', img)
            np.save(save_label + str(save_patch_name) + '.npy', mask)

            print(patient_name, '  no_tumor first slice:', index_first,
                  '  save slice number name:', tumor_slice_name,
                  '  total slice :', z, '  save patch name:', save_patch_name)
            save_patch_name += 1
            tumor_slice_name += 1

        ##保存没有tumor的后面部分
        for i_n in range(no_tumor_number):
            index_finally = pandding + finally_slice[i_n]
            index = index_finally
            img = pad_img[:, :, index - pandding:index + pandding + 1]
            mask = pad_label[:, :, index - pandding:index + pandding + 1].astype(np.int64)

            np.save(save_image + str(save_patch_name) + '.npy', img)
            np.save(save_label + str(save_patch_name) + '.npy', mask)

            print(patient_name, '  no_tumor finally slice:', index_finally,
                  '  save slice number name:', tumor_slice_name,
                  '  total slice :', z, '  save patch name:', save_patch_name)
            save_patch_name += 1
            tumor_slice_name += 1
    return save_patch_name, tumor_slice_name

def get_3D_patch_slice_augm(data_image, data_masks, patch_size,crop_size,
             save_patch_name,tumor_slice_name, output_path,patient_name):
    save_image = output_path
    save_label = output_path.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    image = crop(data_image,crop_size)
    label = crop(data_masks,crop_size)

    print(image.shape)
    x,y,z = image.shape
    pandding = int((patch_size - 1) / 2)

    for i in range(z):
        if label[ :, :, i].max() == 1:
            index = pandding+i
            img = image[:,:,index-pandding:index+pandding+1]
            mask = label[:,:,index].astype(np.int64)
            np.save(save_image + str(save_patch_name) + '.npy', img)
            np.save(save_label + str(save_patch_name) + '.npy', mask)
            print(patient_name, '  include tumor slice:', i,
                  '  save slice number name:', tumor_slice_name,
                  '  total slice :', z, '  save name patch:', save_patch_name)
            save_patch_name += 1
            tumor_slice_name += 1
    return save_patch_name,tumor_slice_name


def get_3D_patch_slice_test(data_image, data_masks, patch_size, crop_size,
                            save_patch_name, tumor_slice_name, output_path, patient_name, File):
    save_image = output_path + patient_name + '/images/'
    save_label = save_image.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    if File == 'val6':
        # 若为6个病人，则需要转换轴
        data_image = data_image.transpose(1, 0, 2)
        data_masks = data_masks.transpose(1, 0, 2)

    image = crop(data_image, crop_size)
    label = crop(data_masks, crop_size)

    # image = data_image
    # label = data_masks

    print(image.shape)
    x, y, z = image.shape
    pandding = int((patch_size - 1) / 2)
    pad_img, pad_label = pandding_3D(image, label, patch_size, pandding)

    for i in range(z):
        index = pandding + i
        img = pad_img[:, :, index - pandding:index + pandding + 1]
        mask = pad_label[:, :, index - pandding:index + pandding + 1].astype(np.int64)
        # mask = np.expand_dims(mask,axis=-1)
        np.save(save_image + str(save_patch_name) + '.npy', img)
        np.save(save_label + str(save_patch_name) + '.npy', mask)
        print(patient_name, '  include tumor slice:', i,
              '  save slice number name:', tumor_slice_name,
              '  total slice :', z, '  save name patch:', save_patch_name)
        save_patch_name += 1
        tumor_slice_name += 1


if __name__ == "__main__":
    '''
    python get_3D_data.py --augm 4 --patch_size 3
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=0)
    # parser.add_argument('--augm', type=int, default=0)
    args = parser.parse_args()

    name_list = ['1GMQX2WE', '2013543', '2015679', '2015898', '2016042', 'LHJNEWUF']
    train_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'train_80.txt'))
    val6_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'val6.txt'))
    val_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'val.txt'))
    lung_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'train_data.txt'))
    

    main_path = '/media/xd/date/muzhaoshan/LIDC-IDRI data/'
    input_path_80 = main_path + 'LungNoduleLIDCTop100/01subsetUR/'
    input_path_6 = main_path + 'LungSegData/02Normalized_subsetUR/'

    output_main = '/media/xd/date/muzhaoshan/MCM_Transformer/data/'

    patch_size = args.patch_size
    crop_size = args.crop_size
    print('patch size',patch_size)
    if patch_size==0:
        print('error show type!')
        import sys
        sys.exit(0)

    patch_type = 'crop_'+str(crop_size)+'_patch_' + str(patch_size)

    if patch_size == 1:
        train_val_num = 4
    else:
        train_val_num = 4
    print('file number:',train_val_num)
    for Num in range(3):
        if Num == 0:
            select_name = 'train'
            File_list = train_list
            File_leng = len(train_list)
            File = 'train'
            input_path = input_path_80
            output_path = output_main + 'train_data/' + patch_type + '/images/'
        elif Num == 1:
            select_name = 'test'
            File_list = val6_list
            File_leng = len(val6_list)
            File = 'val6'
            input_path = input_path_6
            output_path = output_main + 'test_data/' + patch_type + '/'
        elif Num == 2:
            select_name = 'test'
            File_list = val_list
            File_leng = len(val_list)
            File = 'val'
            input_path = input_path_80
            output_path = output_main + 'test_data/' + patch_type + '/'
        elif Num == 3:
            select_name = 'test'
            File_list = lung_list
            File_leng = len(lung_list)
            File = 'lung'
            input_path = input_path_80
            output_path = output_main + 'test_data/' + patch_type + '/'

        save_patch_name = 0
        tumor_slice_name = 0
        slice_num = 0
        for i in range(File_leng):
            print(save_patch_name, tumor_slice_name)
            if select_name == 'train' or select_name == 'val_test':
                patient_name = File_list[i]
            elif select_name == 'test':
                patient_name = File_list[i]
            # augm的字典路径（使用monai）
            data_image_str = input_path + patient_name + '.npy'
            data_masks_str = input_path.replace('01subsetUR', '01grdtUR') + patient_name + '.npy'
            data_dicts = {'image': data_image_str, 'label': data_masks_str}
            # 读取原始数据
            data_image = np.load(data_image_str)
            data_masks = np.load(data_masks_str)
        #     slice_num += data_masks.shape[-1]
        # print(slice_num)
            # '''
            if select_name == 'train':
                # 原始数据分割patch
                print('----------------------train save raw----------------------')
                save_patch_name, tumor_slice_name = get_3D_patch_slice(data_image, data_masks, patch_size,
                                                                             crop_size,
                                                                             save_patch_name, tumor_slice_name,
                                                                             output_path, patient_name)

                # augm数据分割patch

            #     print('----------------------train save augm flip----------------------')
                # data_augm_image, data_augm_masks = Augm_flip(data_dicts)
            #     save_patch_name, tumor_slice_name = \
            #         get_3D_patch_slice(data_augm_image, data_augm_masks, patch_size, crop_size,
            #                                 save_patch_name, tumor_slice_name, output_path, patient_name)

            #     print('----------------------train save augm rotate----------------------')
            #     data_augm_image, data_augm_masks = Augm_rotate(data_dicts)
            #     save_patch_name, tumor_slice_name = \
            #         get_3D_patch_slice(data_augm_image, data_augm_masks, patch_size, crop_size,
            #                                 save_patch_name, tumor_slice_name, output_path, patient_name)

            #     print('----------------------train save augm spatial----------------------')
            #     data_augm_image, data_augm_masks = Augm_Rand3DElasticd(data_dicts)
            #     save_patch_name, tumor_slice_name = \
            #         get_3D_patch_slice(data_augm_image, data_augm_masks, patch_size, crop_size,
            #                                 save_patch_name, tumor_slice_name, output_path, patient_name)

            #     #print('----------------------train save augm RandGaussianNoise----------------------')
            #     #data_augm_image, data_augm_masks = Augm_RandGaussianNoise(data_dicts)
            #     #save_patch_name, tumor_slice_name = \
            #     #    get_3D_patch_slice_augm(data_augm_image, data_masks, patch_size, crop_size,
            #     #                           save_patch_name, tumor_slice_name, output_path, patient_name)

            #     #print('----------------------train save augm RandGaussianSharpen----------------------')
            #     #data_augm_image, data_augm_masks = Augm_RandGaussianSharpen(data_dicts)
            #     #save_patch_name, tumor_slice_name = \
            #     #    get_3D_patch_slice_augm(data_augm_image, data_masks, patch_size, crop_size,
            #     #                            save_patch_name, tumor_slice_name, output_path, patient_name)

            #     print('----------------------train save augm RandAffined----------------------')
                # data_augm_image, data_augm_masks = Augm_RandAffined(data_dicts)
            #     save_patch_name, tumor_slice_name = \
            #         get_3D_patch_slice(data_augm_image, data_augm_masks, patch_size, crop_size,
            #                                 save_patch_name, tumor_slice_name, output_path, patient_name)
            #     # '''

            elif select_name == 'test':  # 测试集不需要augm
                print('----------------------test save raw----------------------')
                get_3D_patch_slice_test(data_image, data_masks, patch_size, crop_size,
                                        save_patch_name, tumor_slice_name, output_path, patient_name, File)





