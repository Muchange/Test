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
from utils.metrics import Index,dice,CrossEntropyLoss,Diceloss,Focalloss
from torch.utils.data import Dataset, DataLoader
from utils.dataset_augm import *
import glob
from models.model_ import MCM_Hit_
from models.model import MCM_Hit_mor
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
parser.add_argument('--dataset_type', type=str, default=' ')
parser.add_argument('--loss_type', type=str, default='Dice')
parser.add_argument('--window_size', type=int, default=0)
parser.add_argument('--drop', type=float, default=0)
args = parser.parse_args()
'''
python test_single_each_epoch.py --patient_type tumor --loss_type CE+Dice_config_0 --dataset_type nodule --patch_size 3 --window_size 8
'''
criterion_ce = CrossEntropyLoss()
criterion_dice = Diceloss()
criterion_fl =Focalloss()

Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
main_data_path = '/media/xd/date/muzhaoshan/LIDC-IDRI data/'
main_path = '/media/xd/date/muzhaoshan/MCM_Hit_Mor/data/'
patient_type = args.patient_type
if patient_type=='tumor':
    Patient_list = ['1GMQX2WE','2013543','2015679','2015898','2016042','LHJNEWUF']
elif patient_type=='nodule':
    Patient_list = ['LIDC-IDRI-0057','LIDC-IDRI-0094','LIDC-IDRI-0172','LIDC-IDRI-0187',
                    'LIDC-IDRI-0188','LIDC-IDRI-0195','LIDC-IDRI-0865','LIDC-IDRI-0976']
elif patient_type=='train_test':
    Patient_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'train_test.txt'))
    
elif patient_type=='lung_test':
    Patient_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'lung_test.txt'))
elif patient_type=='train_data':
    Patient_list = load_file_name_list(os.path.join('/media/xd/date/muzhaoshan/Unet/', 'train_data.txt'))
else:
    print('error patient type!')
    sys.exit(0)
dataset_type = args.dataset_type
patch_size = args.patch_size
print('patch size',patch_size)
print('patch size',dataset_type)
if patch_size==0:
    print('error show type!')
    import sys
    sys.exit(0)
window_size = (args.window_size,args.window_size)
if window_size[1]==0:
    print('error window size!')
    import sys
    sys.exit(0)
drop_rate  = args.drop

in_chan = patch_size
data_type = 'crop_256_patch_'+str(in_chan)
#data_type = 'agum_crop_double_spatial_256_raw'
loss_type = args.loss_type
save_path = main_path + 'test_result/save_each_epoch/'+dataset_type+'/' +patient_type+'/'+data_type\
            +'_ws_'+str(args.window_size)+'_'+loss_type+'/'
#save_path = main_path + 'test_result/save_each_epoch/'+ patient_type+'/'+'agum_crop_double_spatial_256_raw'+'_'+loss_type+'/'
if not os.path.exists(save_path):
        os.makedirs(save_path)

weigth_input_path = main_path +'/work_dirs/weight/'+dataset_type+'_'+data_type+'_save_each_eopch_'+'ws_'+str(args.window_size)\
                    +'_'+loss_type+'/'
#weigth_input_path = main_path+'work_dirs/weight/'+'agum_crop_double_spatial_256'+'_save_each_epoch_'+loss_type+'/'
logger_path = main_path + 'test_result/save_each_epoch/logger/'+dataset_type+'/'+data_type+'_'+loss_type+'/'

if not os.path.exists(logger_path):
    os.makedirs(logger_path)

logger = get_logger(logger_path+'single_patient_'+str(Time)+'.log')
num_workers = 64
#for j in range(len(Patient_list)):
for j in range(len(Patient_list)):
    name_num = j
    max_dice = 0
    best_epoch = 0
    best_index = []
    log_dict = {'epoch': [],'dice': [],'loss_ce': [],'loss_dice': [],'loss_fl': [],'loss': []}
    logger.info(Patient_list[name_num])
    test_path = main_data_path +'test_data/'+data_type+'/'+ Patient_list[name_num]
    logger.info('data input patch:{}'.format(test_path))
    # test_set = load_data(test_path)
    # test_loader = torch.utils.data.DataLoader(dataset=test_set,
    #                             batch_size=32,
    #                             shuffle=False,num_workers=num_workers)
    test_set =  load_data(test_path,'test',
                                transform=transforms.Compose(
                                            [RandomGenerator()]))
    test_loader = DataLoader(test_set, batch_size=32,shuffle=False,)

    #raw_data = np.load('/media/xd/date/muzhaoshan/patient_data_6/'+Patient_list[name_num]+'.npy')
    #raw_data = crop(raw_data,256)
    for i in range(1,len(os.listdir(weigth_input_path))):
    #for i in range(1,58):
        Dice = 0
        Acc = 0
        Iou = 0
        f_score = 0
        precision = 0
        recall = 0
        loss_ce = 0
        loss_dice = 0
        loss = 0

        logger.info('epoch {}'.format(i))

        weigth_path = weigth_input_path + str(i)+'.pth'
        device = torch.device("cuda")
        model = MCM_Hit_mor(n_channels=in_chan, n_classes=1, window_size = window_size).to(device)

        device = torch.device("cuda")
        model = nn.DataParallel(model, device_ids=[0]).cuda()
        model.to(device)
        checkpoint = torch.load(weigth_path)  
        model.load_state_dict(checkpoint['model'])
        #model.load_state_dict(torch.load(weigth_path,map_location=device))   
        model.to(device)
        model.eval()
            
        data_label = []
        data_pred = []

        with torch.no_grad():
            for image, label in tqdm(test_loader):
                image, label = image.float(), label.float()
                image = image.to(device)
                label = label.to(device)
                # print(image.shape,label.shape)
                
                Output = model(image)
                #print(torch.unique(Output))
                #Output[Output >=0.5] = 1
                #Output[Output <0.5] = 0

                #loss
                #print(Output.shape)
                #loss_ce += criterion_ce(Output, label)
                #loss_dice += criterion_dice(Output, label)
                #loss += criterion_ce(Output, label)+criterion_dice(Output, label)


                #Output = Output.detach().cpu().numpy()
                #label = label.detach().cpu().numpy()

                slice_pred = torch.squeeze(Output,axis = 1)
                #slice_pred = np.squeeze(slice_pred,axis = 0)
                slice_masks = torch.squeeze(label,axis = 1)
                #slice_masks = np.squeeze(slice_masks,axis = 0)
                
                data_pred.append(slice_pred)
                data_label.append(slice_masks)

        pred = torch.vstack(data_pred)
        pred = pred.permute(1,2,0)
        masks = torch.vstack(data_label)
        masks = masks.permute(1,2,0)
        print('pred shape {},masks shape {}'.format(pred.shape,masks.shape))

        #print(torch.unique(pred))
        #print(torch.unique(masks))
        loss_ce = criterion_ce(pred, masks).cpu().numpy()
        loss_dice = criterion_dice(pred, masks).cpu().numpy()
        #loss_fl = criterion_fl(pred, masks).cpu().numpy()
        loss = loss_ce + loss_dice

        pred = pred.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        pred[pred>=0.5]=1
        pred[pred<0.5]=0
        #print(np.unique(pred),np.unique(masks))
        index = Index(pred, masks)
        Dice = round(index[0],6)
        Acc = round(index[1],6)
        Iou = round(index[2],6)
        f_score = round(index[3],6)
        precision = round(index[4],6)
        recall = round(index[5],6)

        #print('verify dice:{:.6f}'.format(dice(pred, raw_data)))
        logger.info('dice:{:.6f}, acc:{:.6f}, Iou:{:.6f}, F_score:{:.6f}, Precision:{:.6f},Recall:{:.6f},'
                    'Loss_ce:{:.6f}, Loss_dice:{:.6f},Loss:{:.6f}'\
                    .format(Dice,Acc, Iou, f_score,precision,recall,loss_ce,loss_dice,loss))
        #logger.info('dice:{:.6f}, acc:{:.6f}, Iou:{:.6f}, F_score:{:.6f}, Precision:{:.6f},Recall:{:.6f},'
        #            'Loss_ce:{:.6f}'\
        #            .format(Dice,Acc, Iou, f_score,precision,recall,loss))

        #save best dice
        if max_dice < Dice:
            max_dice = Dice
            best_index = index
            best_epoch = i
        #save log
        log_dict['dice'].append(Dice)
        log_dict['epoch'].append(i)
        log_dict['loss_ce'].append(loss_ce)
        log_dict['loss_dice'].append(loss_dice)
        #log_dict['loss_fl'].append(loss_fl)
        log_dict['loss'].append(loss)
        
    logger.info('best dice epoch {}'.format(best_epoch))
    logger.info('best dice {}'.format(round(best_index[0],6)))
    print('\n')
    
    save_csv = pd.DataFrame({'patient':Patient_list[name_num],'Dice':round(best_index[0],6),
                            'ACC':round(best_index[1],6),'Iou':round(best_index[2],6),
                            'F_score':round(best_index[3],6),'Precision':round(best_index[4],6),
                            'Recall':round(best_index[5],6),'Best_dice_epoch':[best_epoch]})
    save_csv.to_csv(save_path+Patient_list[name_num]+'.csv',index=False,sep=',')
    plt.figure()
    ##'''
    plt.plot(log_dict['epoch'],log_dict['dice'],\
            log_dict['epoch'],log_dict['loss_ce'],\
            log_dict['epoch'],log_dict['loss_dice'], \
             #log_dict['epoch'], log_dict['loss_fl'], \
             log_dict['epoch'],log_dict['loss'])
    ##'''
    #plt.plot(log_dict['epoch'],log_dict['dice'],\
    #        log_dict['epoch'],log_dict['loss'])
    plt_label =['each_epoch_dice','loss_ce','loss_dice','loss']
    plt.legend(plt_label,loc='best')
    plt.savefig(save_path+Patient_list[name_num]+'.jpg')


csv_path = save_path
all_files = []
for i in range(len(Patient_list)):
    all_files.append(csv_path+Patient_list[i]+'.csv')
df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged = pd.concat(df_from_each_file, ignore_index=True)
Avg=[]
inde = ['Dice','ACC','Iou','F_score','Precision','Recall']
for i in range(len(inde)):
    Avg.append(round(df_merged[inde[i]].mean(),6))
Avg.insert(0,"Avg")
Avg.append("")
avg_csv = pd.DataFrame({'Avg':Avg})
df_merged.loc[len(df_merged)] = Avg
df_merged.to_csv(csv_path+'all_patient.csv')
    

