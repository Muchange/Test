from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
from scipy.ndimage import zoom



def pandding_3D(image, label, patch_size, pandding):
    w, h, z = image.shape
    pad_imgs = np.empty((w, h, z + pandding * 2), dtype=image.dtype)
    pad_label = np.empty((w, h, z + pandding * 2), dtype=label.dtype)
    print('panding shape',pad_imgs.shape)
    start_z = pandding
    end_z = z + pandding

    pad_imgs[:, :, start_z:end_z] = image
    pad_label[:, :, start_z:end_z] = label

    pad_imgs[:, :, :start_z] = image[:, :, :start_z]
    pad_imgs[:, :, end_z + patch_size:] = image[:, :, end_z:]

    pad_label[:, :, :start_z] = label[:, :, :start_z]
    pad_label[:, :, end_z + patch_size:] = label[:, :, end_z:]

    return pad_imgs, pad_label

def get_3D_patch_slice(data_image, data_masks, patch_size,
                       save_patch_name, tumor_slice_name, output_path, patient_name):
    save_image = output_path
    save_label = output_path.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    print('raw shape',data_image.shape)
    image = data_image
    label = data_masks

    # plt.figure()
    x, y, z = image.shape
    pandding = int((patch_size - 1) / 2)
    pad_img,pad_mask = pandding_3D(image, label, patch_size, pandding)
    for i in range(z):
        index = pandding + i
        img = pad_img[:, :, index - pandding:index + pandding + 1]
        mask = pad_mask[:, :, index].astype(np.int64)
        np.save(save_image + str(save_patch_name) + '.npy', img)
        np.save(save_label + str(save_patch_name) + '.npy', mask)
        print(patient_name, '  include tumor slice:', i,
                '  save slice number name:', tumor_slice_name,
                '  total slice :', z, '  save name patch:', save_patch_name)
        save_patch_name += 1
        tumor_slice_name += 1
    return save_patch_name, tumor_slice_name

def get_3D_patch_slice_test(data_image, data_masks, patch_size,
                             save_patch_name, tumor_slice_name, output_path, patient_name):
    save_image = output_path + patient_name + '/images/'
    save_label = save_image.replace('images', 'masks')
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_label):
        os.makedirs(save_label)

    image = data_image
    label = data_masks
    print('raw shape',image.shape)
    # plt.figure()
    x, y, z = image.shape
    pandding = int((patch_size - 1) / 2)
    pad_img, pad_mask = pandding_3D(image, label, patch_size, pandding)
    for i in range(z):
        index = pandding + i
        img = pad_img[:, :, index - pandding:index + pandding + 1]
        mask = pad_mask[:, :, index].astype(np.int64)
        np.save(save_image + str(save_patch_name) + '.npy', img)
        np.save(save_label + str(save_patch_name) + '.npy', mask)
        print(patient_name, '  include tumor slice:', i,
                '  save slice number name:', tumor_slice_name,
                '  total slice :', z, '  save name patch:', save_patch_name)
        save_patch_name += 1
        tumor_slice_name += 1

if __name__ == "__main__":
    '''
    python get_3D_data_sy.py --crop_size 256 --patch_size 3
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=0)
    args = parser.parse_args()

    train_list = open(os.path.join('/media/xd/date/muzhaoshan/Synapse data/RawData/', 'train.txt')).readlines()
    val_list = open(os.path.join('/media/xd/date/muzhaoshan/Synapse data/RawData/', 'valid.txt')).readlines()
    test_list = open(os.path.join('/media/xd/date/muzhaoshan/Synapse data/RawData/', 'test.txt')).readlines()

    main_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/'
    patch_size = args.patch_size
    print('patch size',patch_size)
    crop_size = args.crop_size
    print('crop size',crop_size)
    if patch_size==0 or crop_size==0:
        print('error type!')
        import sys
        sys.exit(0)
    patch_type = 'crop_'+str(crop_size)+'_patch_' + str(patch_size)
    for Num in range(3):
        if Num == 0:
            select_name = 'train'
            File_list = train_list
            File_leng = len(train_list)
            File = 'train'
            input_path_ED = main_path + 're_nor/images/'
            input_path_ES = main_path + 're_nor/images/'
            output_path = main_path + 'train_data/' + patch_type + '/images/'
        elif Num == 1:
            select_name = 'val'
            File_list = test_list
            File_leng = len(test_list)
            input_path_ED = main_path + 're_nor/images/'
            input_path_ES = main_path + 're_nor/images/'
            output_path = main_path + 'valid_data/'
        elif Num == 2:
            select_name = 'test'
            File_list = test_list
            File_leng = len(test_list)
            input_path_ED = main_path + 're_nor/images/'
            input_path_ES = main_path + 're_nor/images/'
            output_path = main_path + 'test_data/'
        save_patch_name = 0
        tumor_slice_name = 0
        for i in range(File_leng):
            patient_name = File_list[i].strip('\n')
            print(patient_name)
            # augm的字典路径（使用monai）
            data_image_str_ED = input_path_ED + patient_name + '.npy'
            data_masks_str_ED = input_path_ED.replace('images', 'masks')+ patient_name + '.npy'

            data_image_str_ES = input_path_ES + patient_name + '.npy'
            data_masks_str_ES = input_path_ES.replace('images', 'masks') + patient_name + '.npy'


            # 读取原始数据
            data_image_ED = np.load(data_image_str_ED)
            data_masks_ED = np.load(data_masks_str_ED)

            data_image_ES = np.load(data_image_str_ES)
            data_masks_ES = np.load(data_masks_str_ES)

            if select_name == 'train':
                # 原始数据分割patch
                print('----------------------train save raw----------------------')
                save_patch_name, tumor_slice_name = get_3D_patch_slice(data_image_ED, data_masks_ED, patch_size,
                                                                        save_patch_name, tumor_slice_name, 
                                                                        output_path, patient_name)
                save_patch_name, tumor_slice_name = get_3D_patch_slice(data_image_ES, data_masks_ES, patch_size,
                                                                       save_patch_name, tumor_slice_name,
                                                                       output_path, patient_name)
            elif select_name == 'val' or select_name == 'test':  # 测试集不需要augm
                print('----------------------test save raw----------------------')
                get_3D_patch_slice_test(data_image_ED, data_masks_ED,patch_size,
                                        save_patch_name, tumor_slice_name, 
                                        output_path, patient_name)
                get_3D_patch_slice_test(data_image_ES, data_masks_ES, patch_size,
                                        save_patch_name, tumor_slice_name,
                                        output_path, patient_name)




