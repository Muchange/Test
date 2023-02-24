import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt

main_path = 'D:/Medical Imaging/ACDC/ACDC/train/'
name = 'case_001_sliceED_9.npz'
data = np.load(main_path + name)
img = data['img']
label = data['label']
# print(img.shape,np.unique(img))
# print(label .shape,np.unique(label))
print(img.min(),img.max())

image_data = sitk.Image(sitk.ReadImage('D:/Medical Imaging/ACDC/ACDC_data/patient001/patient001_frame01.nii.gz'))
image = sitk.GetArrayFromImage(image_data).astype(float)
print(image.shape,np.unique(image))
# image -= image.mean()
# image /= image.std()
image = (image-image.min())/(image.max()-image.min())
print(image.shape,np.unique(image))
print(image.min(),image.max())
print(label .shape,np.unique(label))

# plt.figure()
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(img, cmap='gray')
#
# plt.figure()
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(label, cmap='gray')
# plt.show()