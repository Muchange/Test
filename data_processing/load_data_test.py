import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt


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

main_path = '/media/xd/date/muzhaoshan/Synapse data/author_processing_data/images/'
test_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Synapse data/RawData/', 'Sy_test_.txt'))
len_ = []
sum = 0
train_dict = {}
train_list = []
print(train_dict)
for i in range(len(test_list)):
    name = test_list[i]
    img = np.load(main_path + name + '.npy')
    la = np.load(main_path.replace('images','masks') + name + '.npy')
    index = 0
    for j in range(la.shape[-1]):
        if la[:,:,j].max()>1:
            index = j
            break;
    plt.figure()
    for i in range(0, 16):
        num = index + i
        x = img[:, :,num]
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x, cmap='gray')
    plt.figure()
    for i in range(0, 16):
        num = index + i
        x = la[:, :,num]
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x, cmap='gray')
    plt.show()
    