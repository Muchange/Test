import numpy as np
import SimpleITK as sitk
import os
import scipy.io as scio
import h5py
import matplotlib.pyplot as plt 
from scipy.ndimage import zoom

main_path = 'D:/Medical Imaging/Synapse data/RawData/Training/label/label0001.nii.gz'
save_path = 'D:/Medical Imaging/Synapse data/'

img_data = sitk.Image(sitk.ReadImage(main_path))
# img_data = sitk.GetImageFromArray(image_data)
# img_data.SetSpacing((1, 1, 1))
# print(img_data.GetSize(),img_data.GetSpacing())
img_data = sitk.GetArrayFromImage(img_data)
np.save(save_path+'gt0001.npy',img_data)
# MAX = 275.0
# MIN = -125.0
# nor_image = (img_data - MIN) / (MAX - MIN) # min=-1024 %ImageTensor = ImageTensor/max(ImageTensor(:));
# # imgURN = imgURN/4095       #3071-min
# nor_image[nor_image < 0] = 0
# nor_image[nor_image > 1] = 1
# print(np.unique(nor_image))

# img = nor_image
# img = img.transpose(1, 2, 0)
# img = np.flipud(img)
# print(img.shape)

# main_path = '/media/xd/date/muzhaoshan/Synapse data/Synapse_pro/Synapse/test/case0001.npy.h5'
# f = h5py.File(main_path,'r')
# h5_label = f['label']
# print(np.unique(h5_label,return_counts=True))
# print(h5_label.shape)
# h5_image = f['image']
# img_itk = sitk.GetImageFromArray(h5_image[:].astype(np.float32))
# h5_image = np.array(h5_image[:])
# h5_image = h5_image.transpose(2, 1, 0)
# # h5_image = np.flipud(h5_image)
# zoom_img = zoom(h5_image, (224/512,224/512,1), order=0)
# print(h5_image.shape,zoom_img.shape)
# print((h5_image==img).all())
# print(img_itk.GetSpacing())
# print(np.unique(img),np.unique(h5_image))
# plt.figure()
# for i in range(0, 16):
#     num = i
#     x = img[:, :,num]
#     plt.subplot(4, 4, i + 1)
#     plt.axis('off')
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x, cmap='gray')
# plt.figure()
# for i in range(0, 16):
#     num = i
#     x = h5_image[:, :,num]
#     plt.subplot(4, 4, i + 1)
#     plt.axis('off')
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x, cmap='gray')
# plt.figure()
# for i in range(0, 16):
#     num = i
#     x = zoom_img[:, :,num]
#     plt.subplot(4, 4, i + 1)
#     plt.axis('off')
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x, cmap='gray')
# plt.show()