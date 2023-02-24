import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt

main_path = '/media/xd/date/muzhaoshan/Synapse data/author_processing_data/masks/'
test_list = open(os.path.join('/media/xd/date/muzhaoshan/Synapse data/RawData/', 'Sy_test_.txt')).readlines()
len_ = []
sum = 0
train_dict = {}
train_list = []
print(train_dict)
for i in range(len(test_list)):
    # print(file[i])
    # raw_data = sitk.Image(sitk.ReadImage(main_path + file[i]))
    # raw_size = raw_data.GetSize()
    # raw_spacing = raw_data.GetSpacing()
    # print(raw_size,raw_spacing)
    # data = np.load(main_path_ + file_[i])
    # print(data.shape)
    # assert data.shape[0] == data.shape[1],'error'
    # if data.shape[0] != data.shape[1]:
    #     print('------------------')
    data = np.load(main_path + test_list[i] + '.npy')
    x,y,z = data.shape
    print(test_list[i],z)
    sum +=z
print(sum)
    