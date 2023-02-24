import cv2
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk

def vis_save(original_img, pred):
    # blue = [30, 144, 255]  # aorta
    # green = [0, 255, 0]  # gallbladder
    # red = [255, 0, 0]  # left kidney
    # cyan = [0, 255, 255]  # right kidney
    # pink = [255, 0, 255]  # liver
    # yellow = [255, 255, 0]  # pancreas
    # purple = [128, 0, 255]  # spleen
    # orange = [255, 128, 0]  # stomach

    color = [[30,144,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[128,0,255],[255,128,0]]


    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('./img.png', original_img)
    for i in range(len(color)):
        original_img = np.where(pred == i+1, np.full_like(original_img, color[i]), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./color_img.png', original_img)
    return original_img


main_path = 'D:/Medical Imaging/Synapse data/img0001.npy'
save_path = 'D:/Medical Imaging/Synapse data/'

img = np.load(main_path)
gt = np.load(main_path.replace('img','label'))

img = img[:,:,120].astype(np.float32)
gt = gt[:,:,120].astype(np.float32)
print(img.shape,gt.shape)
x = vis_save(img, gt)

# plt.figure()
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(x)
# plt.show()
