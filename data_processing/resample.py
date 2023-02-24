import SimpleITK as sitk
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
"""
resample
"""


def resampleVolume(re_spacing, raw_data,is_label = False):
    """
    将体数据重采样的指定的spacing大小
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    raw_data：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]

    # 读取文件的size和spacing信息
    raw_size = raw_data.GetSize()
    raw_spacing = raw_data.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(raw_size[0] * (raw_spacing[0] / re_spacing[0]))
    outsize[1] = int(raw_size[1] * (raw_spacing[1] / re_spacing[1]))
    outsize[2] = int(raw_size[2] * (raw_spacing[2] / re_spacing[2]))

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(raw_data)
    resampler.SetSize(outsize)
    resampler.SetOutputOrigin(raw_data.GetOrigin())
    resampler.SetOutputDirection(raw_data.GetDirection())
    resampler.SetOutputSpacing(re_spacing)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        #resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkFloat64)  # 线性插值用于PET/CT/MRI之类的，保存float32

    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    re_spacing_img = resampler.Execute(raw_data)  # 得到重新采样后的图像
    return re_spacing_img


def main():
    # 读文件
    data = sitk.Image(sitk.ReadImage("D:/Medical Imaging/2Dunet/LungSegData/00grdt/1GMQX2WE-grdt.hdr"))
    x = sitk.GetArrayFromImage(data)
    print(x.max(),x.min())
    # 重采样
    newvol = resampleVolume([1, 1, 1],data,is_label = True)
    x = sitk.GetArrayFromImage(newvol)
    print(x.max(), x.min())
    plt.figure()
    for i in range(0, 16):
        num = i
        y = x[num, :, :]
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y, cmap='gray')
    plt.show()

    # 写文件
    #wriiter = sitk.ImageFileWriter()
    #wriiter.SetFileName("C:/Users/Miss Change/Desktop/resample_sitk.hdr")
    #wriiter.Execute(newvol)


if __name__ == "__main__":
    main()
    '''
    vol = sitk.ReadImage("D:/Medical Imaging/Lung/Task06_Lung/imagesTr/lung_001.nii.gz")
    data = sitk.GetArrayFromImage(vol)
    plt.figure()
    for i in range(0, 16):
        num = i
        x = data[num,:, :]
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x, cmap='gray')
    plt.show()
    print(data.shape, vol.GetSpacing(),np.unique(data))
    dataFile = 'D:/Medical Imaging/2Dunet/LungSegData/01subsetUR/2015898.mat'
    data = scio.loadmat(dataFile)
    print(data['imgUR'].shape,np.unique(data['imgUR']))
'''