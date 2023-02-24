from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random


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
name = []
list_42 = list(load_file_name_list(os.path.join('C:/Users/Miss Change/Desktop/learn/', 'train.txt')))
list_64 = list(load_file_name_list(os.path.join('./', 'data_100.txt')))
for file in list_64:
    if (file not in list_42):
            #print(name_64)
            with open('data_63.txt', 'a+') as f:  # 设置文件对象
                f.write(file)  # 将字符串写入文件中
                f.write('\n')
            name.append(file)
print(len(name))
