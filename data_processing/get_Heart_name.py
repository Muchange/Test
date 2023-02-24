from configparser import Interpolation
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
import cv2


main_path = 'D:/Medical Imaging/Heart/Task02_Heart/imagesTr/'
file_patient_name = sorted(os.listdir(main_path))
total = []
for x in file_patient_name:
    total.append(int(x[3:6]))
test = sorted(random.sample(range(1, 20), 5))
train = [i for i in total if i not in test]
print('------',len(train),len(test))
#判断两个list是否有相同元素
set_c = set(train) & set(test)
list_c = list(set_c)
print(list_c)
name = []
for file in total:
            #print(name_64)
        with open('heart_total.txt', 'a+') as f:  # 设置文件对象
            f.write(str(file).zfill(3))  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))
name = []
for file in train:
            #print(name_64)
        with open('heart_train.txt', 'a+') as f:  # 设置文件对象
            f.write(str(file).zfill(3))  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))
name = []
for file in test:
            #print(name_64)
        with open('heart_test.txt', 'a+') as f:  # 设置文件对象
            f.write(str(file).zfill(3))  # 将字符串写入文件中
            f.write('\n')
        name.append(file)
print(len(name))
