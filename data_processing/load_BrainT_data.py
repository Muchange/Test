import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
import torch

# main_path = 'D:/Medical Imaging/ACDC/ACDC/train/'
# name = 'case_001_sliceED_9.npz'
# data = np.load(main_path + name)
# img = data['img']
# label = data['label']
# # print(img.shape,np.unique(img))
# # print(label .shape,np.unique(label))
# print(img.min(),img.max())

# main_path = 'D:/Medical Imaging/BrainTumour/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz'
file_name = './BT_total.txt'
main_path = 'D:/Medical Imaging/BrainTumour/Task01_BrainTumour/labelsTr/'
file_list = open(os.path.join(file_name)).readlines()
total = 0
for i in range(len(file_list)):
    patient_num = file_list[i].strip('\n')
    image_path = main_path + 'BRATS_'+patient_num+'.nii.gz'
    # print(patient_num)
    # image_data = sitk.Image(sitk.ReadImage(image_path))
    # image = sitk.GetArrayFromImage(image_data).astype(float)
    image_path = main_path + 'BRATS_'+patient_num+'.nii.gz'
    label_data = sitk.Image(sitk.ReadImage(image_path))
    x,y,z = label_data.GetSize()
    print(x,y,z)
    total += z
print(total)
    # print(label_data.GetSize())
    # img = image.transpose(2,3,1,0)
    # print(img.shape)
    # img_torch = torch.from_numpy(img)
    # img_torch = torch.flatten(img_torch, 2,3)
    # img = img_torch.numpy()
    # print(img.shape)
    # show = False
    # if show:
    #     d,z,x,y = image.shape
    #     plt.figure()
    #     show_num = 4
    #     sh_d = int(d ** 0.5)
    #     sh = int(show_num ** 0.5)
    #     for i in range(0,d):
    #         num = i
    #         x = image[num,50,:,:]
    #         plt.subplot(sh_d,sh_d,i+1)
    #         plt.axis('off')
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.imshow(x,cmap='gray')
    #
    #     plt.figure()
    #     for i in range(0,show_num):
    #         num = i + 50
    #         x = image[0,num,:,:]
    #         plt.subplot(sh,sh,i+1)
    #         plt.axis('off')
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.imshow(x,cmap='gray')
    #     plt.figure()
    #     img = image[:,50:53,:,:]
    #     img_torch = torch.from_numpy(img)
    #     img_torch = torch.flatten(img_torch, 0,1)
    #     img = img_torch.numpy()
    #     c,x,y = img.shape
    #     print(img.shape)
    #     sh = int(c ** 0.5) + 1
    #     for i in range(0, c):
    #         num = i
    #         x = img[num, :, :]
    #         plt.subplot(sh, sh, i + 1)
    #         plt.axis('off')
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.imshow(x, cmap='gray')
    #     plt.show()