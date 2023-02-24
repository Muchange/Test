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
from utils.metrics_sy import Train_index_sy,DiceLoss_sy
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from utils.dataset_augm_sy import *
from utils.overlap import *
import glob
from model.unet_model import UNet
import os
import time
from utils.getlogger import get_logger
import argparse
import sys
from torchvision import transforms

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


parser = argparse.ArgumentParser()
parser.add_argument('--patient_type', type=str, default=' ')
parser.add_argument('--patch_size', type=int, default=0)
parser.add_argument('--crop_size', type=int, default=0)
parser.add_argument('--num_classes', type=int,
                    default=14, help='output channel of network')
parser.add_argument('--loss_type', type=str, default='Dice')
parser.add_argument('--dataset_type', type=str, default='noudle')
args = parser.parse_args()

Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

'''
python test_single_each_epoch_sy.py --patient_type tumor --loss_type CE+Dice_config_0 --crop_size 256 --dataset_type Synapse --patch_size 1
'''

#====================================path================================
main_data_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/'
main_path = '/media/xd/date/muzhaoshan/Unet/data/'
patient_type = args.patient_type

#=============================parser args========================
num_classes = args.num_classes
dataset_type = args.dataset_type
patch_size = args.patch_size
print('patch size',patch_size)
crop_size = args.crop_size
print('crop type',crop_size)
if patch_size==0:
    print('error show type!')
    import sys
    sys.exit(0)
in_chan = patch_size
data_type = 'crop_'+str(crop_size)+'_patch_'+str(in_chan)

#data_type = 'agum_crop_double_spatial_256_raw'
loss_type = args.loss_type
#========overlap==========
stride_size = [int(crop_size/2),int(crop_size/2)]
over_size = [int(crop_size),int(crop_size)]
Pad_way = 'zero'

#===========================================file path=================================
save_path = main_path + 'test_result/save_each_epoch/'+dataset_type+'/'+data_type+'_'+loss_type+'/'
#save_path = main_path + 'test_result/save_each_epoch/'+ patient_type+'/'+'agum_crop_double_spatial_256_raw'+'_'+loss_type+'/'
if not os.path.exists(save_path):
        os.makedirs(save_path)

weigth_input_path = main_path+'work_dirs/weight/'+dataset_type+'_'+data_type+'_save_each_eopch_'+loss_type+'/'
#weigth_input_path = main_path+'work_dirs/weight/'+'agum_crop_double_spatial_256'+'_save_each_epoch_'+loss_type+'/'
logger_path = main_path + 'test_result/save_each_epoch/logger/'+dataset_type+'/'+data_type+'_'+loss_type+'/'
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
logger = get_logger(logger_path+'single_patient_'+str(Time)+'.log')

#===============================loss=======================
criterion_ce = CrossEntropyLoss()
criterion_dice = DiceLoss_sy(num_classes)


#======================================test pre=====================
max_dice = 0
best_epoch = 0
best_index = []
log_dict = {'epoch': [],'dice': [],'hd95': [],'loss_ce': [],'loss_dice': [],'loss_fl': [],'loss': []}
test_path = main_data_path +'test_data/'+data_type+'/'
logger.info('data input patch:{}'.format(test_path))


test_set = load_data(test_path,'test',
                        transform=transforms.Compose(
                                    [RandomGenerator()]))
test_loader = DataLoader(test_set, batch_size=1,shuffle=False,
                        num_workers=8)

#======================================testing========================
for i in range(1,len(os.listdir(weigth_input_path))):
#for i in range(1,100):
    metric_list = 0.0
    logger.info('epoch {}'.format(i))

    weigth_path = weigth_input_path + str(i)+'.pth'

#========================model load======================
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda")
    model = UNet(n_channels=in_chan,n_classes=num_classes).to(device)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    model.to(device)
    checkpoint = torch.load(weigth_path)  
    model.load_state_dict(checkpoint['model'])
    #model.load_state_dict(torch.load(weigth_path,map_location=device))   
    model.to(device)
    model.eval()

    Loss_ce = 0
    Loss_dice = 0
    Loss = 0
    test_len = len(test_loader)
    for image, label in tqdm(test_loader):
        with torch.no_grad():
            image, label = image.float(), label.float()
            image = image.to(device)
            label = label.to(device)
            # print(image.shape,label.shape)
            B,raw_H,raw_W = label.shape
            #input:B,C,H,W,B=1
            image = image.squeeze(0).permute(1,2,0)
            image_patch = extract_ordered_pathches(image, stride_size, over_size, pad_way=Pad_way)
            #patch_num,H,W,C = image_patch.shape
            # print('overlap shape:[patch_num,H,W,C',image_patch.shape)
            # plt.figure()
            # Size = math.ceil(image_patch.shape[0] ** 0.5)
            # for i in range(image_patch.shape[0]):
            #     plt.subplot(Size, Size, i + 1)
            #     plt.imshow(image_patch[i, :, :, 1], cmap='gray')
            #     plt.xticks([])
            #     plt.yticks([])
            # plt.show()

            image_patch = image_patch.permute(0,3,1,2)
            patch_num,C,H,W = image_patch.shape
            # print('overlap shape:[patch_num,C,H,W',image_patch.shape)
            output_patch = []
            for i in range(patch_num):
                image = image_patch[i,:]
                image = image.unsqueeze(0) #B,C,H,W,B=1
                output_ = model(image).permute(0,2,3,1)  #B,H,W,num_class,B=1
                # print(output_.shape)
                output_patch.append(output_)
            output = torch.vstack(output_patch)
            # print(output.shape)
            re_output = rebuild_images(stride_size, over_size, output, raw_img_size = (raw_H,raw_W,num_classes))
            output = re_output.permute(2,0,1).unsqueeze(0).to(device)
            # print(output.shape, label.shape)
            loss_ce = criterion_ce(output, label[:].long())
            loss_dice = criterion_dice(output, label, softmax=True)
            Loss_ce += loss_ce.item()
            Loss_dice += loss_dice.item()
            Loss = Loss_ce+Loss_dice
            metric_ = Train_index_sy(output, label,num_classes)
            metric_list += np.array(metric_)
    metric_list = metric_list / test_len
    Loss_ce = Loss_ce/test_len
    Loss_dice = Loss_dice / test_len
    Loss = Loss / test_len
    for i in range(num_classes):
        logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    Index = np.mean(metric_list, axis=0)

    Dice = round(Index[0],6)
    Hd95 = round(Index[1],6)
    Acc = round(Index[2],6)
    Iou = round(Index[3],6)
    f_score = round(Index[4],6)
    precision = round(Index[5],6)
    recall = round(Index[6],6)

    #print('verify dice:{:.6f}'.format(dice(pred, raw_data)))
    logger.info('dice:{:.6f}, Hd95:{:.6f},acc:{:.6f}, Iou:{:.6f}, F_score:{:.6f}, Precision:{:.6f},Recall:{:.6f},'
                'Loss_ce:{:.6f}, Loss_dice:{:.6f},Loss:{:.6f}'\
                .format(Dice,Hd95,Acc, Iou, f_score,precision,recall,Loss_ce,Loss_dice,Loss))
    #save best dice
    if max_dice < Dice:
        max_dice = Dice
        best_index = Index
        best_epoch = i
    #save log
    log_dict['dice'].append(Dice)
    log_dict['hd95'].append(Hd95)
    log_dict['epoch'].append(i)
    log_dict['loss_ce'].append(Loss_ce)
    log_dict['loss_dice'].append(Loss_dice)
    log_dict['loss'].append(Loss)
    
logger.info('best dice epoch {}'.format(best_epoch))
logger.info('best dice {}'.format(round(best_index[0],6)))
print('\n')

save_csv = pd.DataFrame({'patient':Patient_list[name_num],'Dice':round(best_index[0],6),
                            'Hd95':round(best_index[1],6),
                        'ACC':round(best_index[2],6),'Iou':round(best_index[3],6),
                        'F_score':round(best_index[4],6),'Precision':round(best_index[5],6),
                        'Recall':round(best_index[6],6),'Best_dice_epoch':[best_epoch]})
save_csv.to_csv(save_path+Patient_list[name_num]+'.csv',index=False,sep=',')
plt.figure()
##'''
plt.plot(log_dict['epoch'], log_dict['dice'], \
            log_dict['epoch'], log_dict['hd95'], \
            log_dict['epoch'], log_dict['loss_ce'], \
            log_dict['epoch'], log_dict['loss_dice'], \
            log_dict['epoch'], log_dict['loss'])
##'''
#plt.plot(log_dict['epoch'],log_dict['dice'],\
#        log_dict['epoch'],log_dict['loss'])
plt_label =['dice','hd95','loss_ce','loss_dice','loss']
plt.legend(plt_label,loc='best')
plt.savefig(save_path+Patient_list[name_num]+'.jpg')


csv_path = save_path
all_files = []
for i in range(len(Patient_list)):
    all_files.append(csv_path+Patient_list[i]+'.csv')
df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged = pd.concat(df_from_each_file, ignore_index=True)
Avg=[]
inde = ['Dice','Hd95','ACC','Iou','F_score','Precision','Recall']
for i in range(len(inde)):
    Avg.append(round(df_merged[inde[i]].mean(),6))
Avg.insert(0,"Avg")
Avg.append("")
avg_csv = pd.DataFrame({'Avg':Avg})
df_merged.loc[len(df_merged)] = Avg
df_merged.to_csv(csv_path+'all_patient.csv')


