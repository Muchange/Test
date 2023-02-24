import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
import h5py


main_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/test_data/0001/images/'
file_list = os.listdir(main_path)
file_list.sort(key=lambda x:int(x[:-4]))
# print(file_list)
show_num = 40
plt.figure()
for i in range(36):
    num =  show_num+i
    x = np.load(main_path + file_list[num])
    plt.subplot(6, 6, i + 1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x, cmap='gray')
plt.figure()
for i in range(36):
    num =  show_num+i
    x = np.load(main_path.replace('images','masks') + file_list[num] )
    plt.subplot(6, 6, i + 1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x, cmap='gray')
# plt.show()
main_path = '/media/xd/date/muzhaoshan/Synapse data/Synapse_pro/Synapse/test/case0001.npy.h5'
f = h5py.File(main_path,'r')
h5_image = f['image']
h5_label = f['label']
plt.figure()
for i in range(36):
    num =  show_num+i
    x = h5_image[num,:,:]
    plt.subplot(6, 6, i + 1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x, cmap='gray')
plt.figure()
for i in range(36):
    num =  show_num+i
    x = h5_label[num,:,:]
    plt.subplot(6, 6, i + 1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x, cmap='gray')
plt.show()  