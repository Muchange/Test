import numpy as np
import SimpleITK as sitk
import os
import scipy.io as scio
import h5py
import matplotlib.pyplot as plt 
from scipy.ndimage import zoom

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


def get_dict():
    train_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Synapse data/RawData/', 'Sy_train_.txt'))
    train_dict = {}
    main_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/Training/img/'

    for i in range(len(train_list)):
        img_data = sitk.Image(sitk.ReadImage(main_path + 'img'+train_list[i]+'.nii.gz'))
        data = sitk.GetArrayFromImage(img_data)
        z,x,y = data.shape
        print(train_list[i],z)
        my_temp_dict = {train_list[i]:z}
        train_dict.update(**my_temp_dict) 
    return train_list,train_dict


main_path = '/media/xd/date/muzhaoshan/Synapse data/Synapse_pro/Synapse/'
out_path = '/media/xd/date/muzhaoshan/Synapse data/'
for i in range(2):
    if i ==0:
        train_in = main_path + 'train/'

        output_path_img = out_path + 'author_processing_data/images/'
        output_path_label = output_path_img.replace('images','masks')
        if not os.path.exists(output_path_img):
            os.makedirs(output_path_img)
        if not os.path.exists(output_path_label):
            os.makedirs(output_path_label)

        train_list,train_dict = get_dict()
        for file_ in range(len(train_dict)):
            file_name = train_list[file_]
            file_slice_num = train_dict[train_list[file_]]
            image_slice = []
            label_slice = []
            for file_slice in range(file_slice_num):
                slice_name = str(file_slice)
                slice_name = slice_name.zfill(3)
                in_path = train_in+'case'+file_name+'_'+'slice'+slice_name+'.npz'
                # print(in_path)
                data = np.load(in_path)
                data_image = data['image']
                data_image = np.expand_dims(data_image,axis = 0)
                data_label = data['label']
                data_label = np.expand_dims(data_label,axis = 0)
                # print(data_image.shape,data_label.shape)
                # print(np.unique(data_image),np.unique(data_label))
                image_slice.append(data_image)
                label_slice.append(data_label)
            image_single_file = np.stack(image_slice).squeeze(1).transpose(2,1,0)
            label_single_file = np.stack(label_slice).squeeze(1).transpose(2,1,0)
            print(image_single_file.shape,label_single_file.shape)
            img_data = sitk.GetImageFromArray(image_single_file)
            la_data = sitk.GetImageFromArray(label_single_file)
            print(img_data.GetSpacing(),la_data.GetSpacing())
            show = True
            if show:
                index = 0
                for i in range(label_single_file.shape[-1]):
                    if label_single_file[:,:,i].max() >1:
                        index = i
                        break;
                show_ = []
                show_.append(image_single_file)
                show_.append(label_single_file)
                for i in range(2):
                    plt.figure()
                    for j in range(16):
                        num = index +j 
                        plt.subplot(4, 4, j + 1)
                        plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(show_[i][:,:,num], cmap='gray')
                plt.show()
            # print(np.unique(image_single_file),np.unique(label_single_file))
            # np.save(output_path_img+file_name+'.npy',image_single_file)
            # np.save(output_path_label+file_name+'.npy',label_single_file)
    elif i==1:
        test_in = main_path + 'test/'
        output_path_img = out_path + 'author_processing_data/images/'
        output_path_label = output_path_img.replace('images','masks')

        test_list = np.sort(os.listdir(test_in))
        for file_ in range(len(test_list)):
            file_name = test_list[file_]
            load_path = test_in + file_name
            print(load_path)
            print(file_name[4:8])
            f = h5py.File(load_path,'r')
            image = np.array(f['image'])
            label = np.array(f['label'])
            image = image.transpose(2, 1, 0)
            label = label.transpose(2, 1, 0)
            print(image.shape,label.shape)
            img_data = sitk.GetImageFromArray(image)
            la_data = sitk.GetImageFromArray(label)
            print(img_data.GetSpacing(),la_data.GetSpacing())
            show = False
            if show:
                index = 0
                for i in range(label.shape[-1]):
                    if label[:,:,i].max() >1:
                        index = i
                        break;
                show_ = []
                show_.append(image)
                show_.append(label)
                for i in range(2):
                    plt.figure()
                    for j in range(16):
                        num = index +j 
                        plt.subplot(4, 4, j + 1)
                        plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(show_[i][:,:,num], cmap='gray')
                plt.show()
            np.save(output_path_img+file_name[4:8]+'.npy',image)
            np.save(output_path_label+file_name[4:8]+'.npy',label)

