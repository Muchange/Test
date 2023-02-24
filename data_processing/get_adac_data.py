from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk

file_name = 'train.txt'
main_path = 'D:/Medical Imaging/ADAC/ACAD_data/'
file_list = open(os.path.join(main_path, file_name)).readlines()
for i in range(len(file_list)):
    patient_num = file_list[i].strip('\n')
    patient_path = main_path + 'patient'+patient_num+'/'
    patient_file_name = sorted(os.listdir(patient_path))

    img_ED_path = patient_path + patient_file_name[2]
    gt_ED_path = patient_path + patient_file_name[3]
    img_ES_path = patient_path + patient_file_name[4]
    gt_ES_path = patient_path + patient_file_name[5]

    # x = 'D:/Medical Imaging/ADAC/ACAD_data/patient001/patient001_frame01.nii.gz'
    # print(patient_path)

    img_ED = sitk.Image(sitk.ReadImage(img_ED_path))
    gt_ED = sitk.Image(sitk.ReadImage(gt_ED_path))
    img_ES = sitk.Image(sitk.ReadImage(img_ES_path))
    gt_ES = sitk.Image(sitk.ReadImage(gt_ES_path))

    img_ED_data = sitk.GetArrayFromImage(img_ED)
    gt_ED_data = sitk.GetArrayFromImage(gt_ED)
    img_ES_data = sitk.GetArrayFromImage(img_ES)
    gt_ES_data = sitk.GetArrayFromImage(gt_ES)


    # raw_size = img_ED.GetSize()
    # print(raw_size)
    # raw_size = img_ES.GetSize()
    # print(raw_size)
    raw_spacing = img_ED.GetSpacing()
    print(raw_spacing)
    # raw_spacing = img_ES.GetSpacing()
    # print(raw_spacing)
    # image_max, image_min = img_ES_data.max(), img_ES_data.min()
    # print(image_max,image_min)
    # image_max, image_min = gt_ES_data.max(), gt_ES_data.min()
    # print(image_max, image_min)
    # image_uni, gt_uni = np.unique(img_ES_data), np.unique(gt_ES_data)
    # print(image_uni, gt_uni)
