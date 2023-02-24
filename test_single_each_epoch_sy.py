from operator import mod
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
# from models.model_merge_conv_5_SE_ablation import MCM_Hit_mor
from models.model_merge_conv_5_SE import MCM_Hit_mor
import os
import time
from utils.getlogger import get_logger
import argparse
import sys
from thop import profile
from torchvision import transforms
from scipy.ndimage import zoom
import SimpleITK as sitk
from utils.overlap import *
from monai.inferers import sliding_window_inference


Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

#===============================================init======================================
parser = argparse.ArgumentParser()
parser.add_argument('--patch_size', type=int, default=0)
parser.add_argument('--dataset_type',type=str, default='Synapse')
parser.add_argument('--loss_type', type=str, default='Dice')
parser.add_argument('--num_classes', type=float, default=9)
parser.add_argument('--save_pred', type=str, default='False')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--window_size', type=int, default=8)
parser.add_argument('--load_model_num', type=int, default=10)
args = parser.parse_args()

'''
python test_single_each_epoch_sy.py  --loss_type CE+Dice_config_merge_ablation_conv_5_SE_2 --save_pred True --load_model_num -1 --patch_size 1 
'''

torch.cuda.empty_cache()
#===============================================path======================================
main_data_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/'
main_path = '/media/xd/date/muzhaoshan/MCM_Hit_Mor/data/'
test_path = main_data_path +'test_data/'
test_name_list = open(os.path.join(main_data_path, 'Sy_test.txt')).readlines()


#======================================parser args==========================================
num_classes = args.num_classes
dataset_type = args.dataset_type
patch_size = args.patch_size
loss_type = args.loss_type
load_model_num = args.load_model_num
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
window_size = (args.window_size,args.window_size)
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
logger = get_logger(logger_path+'single_patient_'+str(Time)+'.log')

logger.info(loss_type)

#======================================load traing best dice==========================================
df = pd.read_csv(save_path + dataset_type+'.csv',sep=',')
best_dice = df['Dice'][num_classes - 1]
best_epoch = df['Best_eopch'][0]
if load_model_num == -1:
    load_model_num = 'best_'+str(best_epoch)
else:
    load_model_num = str(load_model_num)

#======================================label and indicator==========================================
# indicator_list=['Dice','Hd95','ACC','Iou','F_score','Precision','Recall']
indicator_list=['Dice','Hd95','ACC','Iou','F_score','Precision','Recall']
# label_list=["Aorta","Gallbladder","Kidney(L)","Kidney(R)","Liver","Pancreas","Spleen","Stomach","Avge"]  #avg
label_list=["Aort","Gall","Kidney(L)","Kidney(R)","Liver","Pancreas","Spleen","Stomach","Avge"]  #avg
# print(len(label_list))

#======================================load model==========================================
weigth_path = weigth_input_path + load_model_num+'.pth'
logger.info(weigth_path)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")
model = MCM_Hit_mor(n_channels=in_chan, n_classes=num_classes , window_size = window_size).to(device)
# print(model)
# model = nn.DataParallel(model, device_ids=[0])
checkpoint = torch.load(weigth_path)
# model.load_state_dict(checkpoint['model'],strict=False)
model.load_state_dict(checkpoint['model'])
model.to(device)

# #======================================params flops==========================================
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# logger.info("Total_params: {} M".format(pytorch_total_params/1000000.0))
# x = torch.randn((4,in_chan,256,256)).cuda()
# flops,params = profile(model,(x,))
# logger.info('profile params:{} M'.format(params/1000000.0))
# logger.info('profile flops:{} M'.format(flops/1000000.0))

model.eval()

#======================================testing==========================================
test_len = len(test_name_list)
metric_list = 0.0
for i in range(test_len):
    # ======================================load data==========================================
    test_name = test_name_list[i].strip('\n')
    test_path_in = test_path + data_type +'/'+test_name + '/'
    # test_path_in = test_path + '/'+data_type + '/' 
    test_set = load_data(test_path_in,image_size, 'test',
                            transform=transforms.Compose(
                                [RandomGenerator()]))
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=1,
                                                shuffle=False, num_workers=8)

    data_label = []
    data_pred = []
    with torch.no_grad():
        for image_input, label_input in tqdm(test_loader):      #single case slice
            image_input, label_input = image_input.to(device).float(), label_input.to(device).float()
            B, C, raw_H, raw_W = image_input.shape
            # prediction = np.zeros_like(label_input) #B,raw_H,raw_W
            ##=========================sliding_window_inference=======================
            output = sliding_window_inference(image_input,(image_size,image_size),8,model,overlap=0.75)
            output_ = torch.argmax(torch.softmax(output, dim=1), dim=1) #1,H,W
            # print(output.shape)
            data_label.append(label_input)     #raw_H,raw_W
            data_pred.append(output_)    #1,raw_H,raw_W
    # =================single case===============
    pred = torch.stack(data_pred).squeeze(1)    #slice,raw_H,raw_W
    mask = torch.stack(data_label).squeeze(1)    #slice,,raw_H,raw_W
    pred = pred.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()
    print(pred.shape,mask.shape)
    # if save_pred:
    #     prd_itk = sitk.GetImageFromArray(pred.astype(np.float32))
    #     sitk.WriteImage(prd_itk, save_path + '/' + test_name + "_pred.nii")
    #     print('save ' + test_name + "_pred.nii")

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
best_metric_list = np.array(best_metric_list)
best_metric_list = np.round(best_metric_list,6)
logger.info('best dice hd95 {}  {}'.format(Index[0], Index[1]))
logger.info('best epoch  {}'.format(best_epoch))

print('\n')
# print(best_metric_list)
if save_pred: 
    save_csv = pd.DataFrame(best_metric_list, columns=indicator_list)
    save_csv.insert(0, 'Class', label_list)
    save_csv.insert(len(indicator_list)+1, 'Best_eopch', best_epoch)
    save_csv.to_csv(save_path+dataset_type+'.csv',index=False,sep=',')







