import pandas as pd
import scipy.io as scio
import math
import scipy.io as sc
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.metrics_sy import Train_index_sy
from torch.utils.data import Dataset, DataLoader
from utils.dataset_synapse_overlap import *
import glob
from model.unet_model import UNet
import os
import time
from utils.getlogger import get_logger
import argparse
import sys
from torchvision import transforms
from scipy.ndimage import zoom
import SimpleITK as sitk
from utils.overlap import *



Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

#===============================================init======================================
parser = argparse.ArgumentParser()
parser.add_argument('--patch_size', type=int, default=0)
parser.add_argument('--dataset_type',type=str, default='Synapse')
parser.add_argument('--loss_type', type=str, default='Dice')
parser.add_argument('--window_size', type=int, default=0)
parser.add_argument('--num_classes', type=float, default=9)
parser.add_argument('--save_pred', type=str, default='False')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--save_model_epoch', type=int, default=5)
args = parser.parse_args()
'''
python test_single_each_epoch_sy.py  --loss_type CE+Dice_config_0 --save_pred False --save_model_epoch 250 --patch_size 1 
'''

torch.cuda.empty_cache()
#===============================================path======================================
main_data_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/'
main_path = '/media/xd/date/muzhaoshan/Unet/data/'
test_path = main_data_path +'test_data/'
test_name_list = open(os.path.join(main_data_path, 'Sy_test.txt')).readlines()

#======================================parser args==========================================
num_classes = args.num_classes
dataset_type = args.dataset_type
patch_size = args.patch_size
loss_type = args.loss_type
save_model_epoch = args.save_model_epoch
print('patch size',patch_size)
if patch_size==0:
    print('error show type!')
    import sys
    sys.exit(0)
if args.save_pred=='True':
    save_pred = 1
elif args.save_pred=='False':
    save_pred = 0
else:
    print('error show type!')
    sys.exit(0)
image_size = args.image_size
overpatch_size= [image_size,image_size]
stride_size = [int(image_size /2), int(image_size /2)]
# stride_size = [image_size,image_size]

#======================================input path==========================================
in_chan = patch_size
data_type = 'crop_'+str(image_size)+'_patch_'+str(in_chan)

save_path = main_path + 'test_result/save_each_epoch/'+ dataset_type+'/'+data_type +'_'+loss_type+'/'
if not os.path.exists(save_path):
        os.makedirs(save_path)

weigth_input_path = main_path +'/work_dirs/weight/'+dataset_type+'_'+data_type+'_save_each_eopch_'+loss_type+'/'

logger_path = main_path + 'test_result/save_each_epoch/logger/'+dataset_type+'_'+data_type+'_'+loss_type+'/'
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
logger = get_logger(logger_path+'single_patient_'+str(Time)+'monai_testing_window.log')

#======================================load traing best dice==========================================
df = pd.read_csv(save_path + dataset_type+'.csv',sep=',')
best_dice = df['Avge'][0]
best_epoch = df['Best_eopch'][0]
#======================================label and indicator==========================================
# indicator_list=['Dice','Hd95','ACC','Iou','F_score','Precision','Recall']
indicator_list=['Dice','Hd95','ACC','Iou','F_score','Precision','Recall']
label_list=["Aorta","Gallbladder","Kidney(L)","Kidney(R)","Liver","Pancreas","Spleen","Stomach","Avge"]  #avg
# print(len(label_list))
#======================================testing==========================================
test_len = len(test_name_list)
metric_list = 0.0
for i in range(test_len):
    # ======================================load data==========================================
    test_name = test_name_list[i].strip('\n')
    test_path_in = test_path + test_name + '/'
    # test_path_in = test_path + '/'+data_type + '/' 
    test_set = load_data(test_path_in,image_size, 'test',
                            transform=transforms.Compose(
                                [RandomGenerator()]))
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=1,
                                                shuffle=True, num_workers=8)


    #======================================load model==========================================
    weigth_path = weigth_input_path + str(save_model_epoch)+'.pth'
    # print(weigth_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda")
    model = UNet(n_channels=in_chan, n_classes=num_classes).to(device)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load(weigth_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    data_label = []
    data_pred = []
    with torch.no_grad():
        for image_input, label_input in tqdm(test_loader):      #single case slice
            image_input, label_input = image_input.float(), label_input.float()
            image_input,label_input = image_input.cpu().detach().numpy(), label_input.cpu().detach().numpy()
            B, C, raw_H, raw_W = image_input.shape
            # prediction = np.zeros_like(label_input) #B,raw_H,raw_W
            image_b = image_input.squeeze(0) #C,H,W
            image_patch = image_b.transpose(1,2,0) #H,W,C
            image_patch = extract_ordered_pathches(image_patch,overpatch_size,stride_size) #num_patch,H,W,C
            image_patch = image_patch.transpose(0, 3, 1, 2)
            patch_num, C, H, W = image_patch.shape

            image_patch = torch.from_numpy(image_patch).float().to(device)
            output_patch = []
            for i in range(patch_num):
                image = image_patch[i, :,:,:] #C,H,W
                image = image.unsqueeze(0)  # B,C,H,W,B=1
                outputs = model(image)
                outputs_ima = outputs.cpu().detach().numpy()
                output_ = torch.argmax(torch.softmax(outputs, dim=1), dim=1).unsqueeze(0) #1,H,W
                output_patch.append(output_)
            output = torch.vstack(output_patch).squeeze(1) #num_path,H,W
            output = output.cpu().detach().numpy()
            output = output.transpose(0,2,3,1)
            re_output = rebuild_images(overpatch_size,stride_size, output, raw_img_size=(raw_H, raw_W,9)) #raw_H,raw_W
            data_label.append(label_input)     #raw_H,raw_W
            data_pred.append(re_output)    #1,raw_H,raw_W
    # =================single case===============
    pred = np.stack(data_pred)     #slice,raw_H,raw_W
    mask = np.stack(data_label).squeeze(1)    #slice,,raw_H,raw_W
    # print(pred.shape,mask.shape)
    if save_pred is not None:
        prd_itk = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(prd_itk, save_path + '/' + test_name + "_pred.nii")

    #=================single indicators===============
    metric_ = Train_index_sy(pred, mask, num_classes,'test')    #single case indicators
    logger.info('case %s mean_dice %f mean_hd95 %f' % (test_name, np.mean(metric_, axis=0)[0],\
                                                                        np.mean(metric_, axis=0)[1]))
    metric_list += np.array(metric_)  # count all case indicators

metric_list = metric_list / test_len  #mean all case indicators
# print(len(metric_list))
# =================mean all case indicators===============
logger.info('total %d case mean'%(test_len))
logger.info('\t\tmean_dice\thd95\t\tmacc\t\tmIou\t\tmF_score\tmPrecision\tmRecall ' )
for i in range(1,num_classes):
    ii = i-1
    logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f' %
                                (label_list[ii][:4], metric_list[ii][0],
                                metric_list[ii][1], metric_list[ii][2],
                                metric_list[ii][3], metric_list[ii][4],
                                metric_list[ii][5], metric_list[ii][6]))
#=================avg mean class(所有病例的平均)===============
Index = np.mean(metric_list, axis=0)
logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f' %
            (label_list[-1], round(Index[0], 6),
                round(Index[1], 6), round(Index[2], 6),
                round(Index[3], 6), round(Index[4], 6),
                round(Index[5], 6), round(Index[6], 6)))
best_metric_list = []
best_metric_list = metric_list.tolist()
best_metric_list.append(Index.tolist())
best_metric_list = np.array(best_metric_list).transpose(1,0)
best_metric_list = np.round(best_metric_list,6)
logger.info('best dice hd95 {}  {}'.format(Index[0], Index[1]))

print('\n')
# print(best_metric_list)
if Index[0] >= best_dice: 
    save_csv = pd.DataFrame(best_metric_list, columns=label_list)
    save_csv.insert(0, 'Indicator', indicator_list)
    save_csv.insert(len(label_list)+1, 'Best_eopch', best_epoch)
    save_csv.to_csv(save_path+dataset_type+'.csv',index=False,sep=',')







