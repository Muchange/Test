import os
import SimpleITK as sitk
main_path  = '/media/xd/date/muzhaoshan/Synapse data/RawData/Testing/img/'
file = sorted(os.listdir(main_path))
slice_num = 0
for i in range(len(file)):
    print(file[i][3:-7])
    image_data = sitk.Image(sitk.ReadImage(main_path + file[i] ))
    #print(image_data.GetSize())
    # slice_num += image_data.GetSize()[-1]
print(len(file),slice_num)