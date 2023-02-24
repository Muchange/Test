from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
from resample import resampleVolume
import glob
import SimpleITK as sitk
from lungmask import mask
from skimage import measure, color

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

def preprossing_data(main_path,save_path,patient_name):
        image_data = sitk.Image(sitk.ReadImage(main_path + patient_name ))
        mask_data = sitk.Image(sitk.ReadImage(main_path.replace('imagesTr','labelsTr') + patient_name ))

        img = sitk.GetArrayFromImage(image_data)
        la = sitk.GetArrayFromImage(mask_data)


        resample_image = resampleVolume([1.0,1.0,1.0],image_data,is_label = False)
        resample_mask = resampleVolume([1.0,1.0,1.0],mask_data,is_label = True)
        label = sitk.GetArrayFromImage(resample_mask)
        label[label < 0.5] = 0
        label[label > 0.5] = 1

        print('raw')
        print(img.max(),img.min())
        print(la.max(),la.min())
        
        #normalized
        
        MAX = 3071
        MIN = -1024
        resample_image = sitk.GetArrayFromImage(resample_image)
        print('resample')
        print(resample_image.max(),resample_image.min())
        print(label.max(),label.min())
        nor_image = (resample_image - MIN) / (MAX-MIN)      #min=-1024 %ImageTensor = ImageTensor/max(ImageTensor(:));
        #imgURN = imgURN/4095       #3071-min
        nor_image[nor_image<0] = 0
        nor_image[nor_image>1] = 1
        #print('\n')

        nor_image = nor_image.transpose(1,2,0)
        nor_image =np.flipud(nor_image)
        label = label.transpose(1,2,0)
        label = np.flipud(label)
        print('resample shape',nor_image.shape,label.shape)
        print('unique',np.unique(nor_image),np.unique(label))
        print('\n')
        '''
        plt.figure()
        slice_index = 165
        for i in range(0,16):
            num = i+slice_index
            x = nor_image[:,:,num]
            plt.subplot(4,4,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')

        plt.figure()
        for i in range(0,16):
            num = i+slice_index
            x = label[:,:,num]
            plt.subplot(4,4,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()
        '''
        # np.save(save_path + patient_name+'.npy',nor_image)
        # np.save(save_path.replace('02Normalized_subsetUR','01grdtUR') + patient_name+'.npy',label)
        return nor_image,label
if __name__ == "__main__":

    lung_train_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'lung_data.txt'))

    main_path = '/media/xd/date/muzhaoshan/MCM_Transformer/data/Lung_data/Task06_Lung/imagesTr/'
    save_path = '/media/xd/date/muzhaoshan/Muzhaoshan/LungNoduleLIDCTop100/02Normalized_subsetUR/'

    for i in range(len(lung_train_list)):
        patient_name = lung_train_list[i]
        print(patient_name)
        preprossing_data(main_path, save_path, patient_name)








