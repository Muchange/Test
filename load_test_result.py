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
from utils.load_dataset import load_data
import glob
from models.model import MCM_Hit
import os
import time
from utils.getlogger import get_logger
import argparse
import sys

'''
op:
python load_test_result.py  --loss_type CE+Dice_config_0 --patient_type tumor --augm 4 --show False --patch_size 3 --window_size 8
'''


parser = argparse.ArgumentParser()
parser.add_argument('--show', type=str, default=' ')
parser.add_argument('--augm', type=str, default=' ')
parser.add_argument('--loss_type', type=str, default=' ')
parser.add_argument('--patient_type', type=str, default=' ')
parser.add_argument('--patch_size', type=int, default=0)
parser.add_argument('--window_size', type=int, default=0)
args = parser.parse_args()
if args.show=='True':
    show = 1
elif args.show=='False':
    show = 0
else:
    print('error show type!')
    sys.exit(0)
window_size = (args.window_size,args.window_size)
if window_size[1]==0:
    print('error window size!')
    import sys
    sys.exit(0)


criterion_ce = CrossEntropyLoss()
criterion_dice = Diceloss()
criterion_fl =Focalloss()

Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
main_data_path = '/media/xd/date/muzhaoshan/MCM_Transformer/data/'
main_path = '/media/xd/date/muzhaoshan/MCM_HIT/data/'
patient_type = args.patient_type
if patient_type=='tumor':
    Patient_list = ['1GMQX2WE','2015679','2015898','2016042','LHJNEWUF']
elif patient_type=='nodule':
    Patient_list = ['LIDC-IDRI-0057','LIDC-IDRI-0094','LIDC-IDRI-0172','LIDC-IDRI-0187',
                    'LIDC-IDRI-0188','LIDC-IDRI-0195','LIDC-IDRI-0865','LIDC-IDRI-0976']
else:
    print('error patient type!')
    sys.exit(0)
patch_size = args.patch_size
augm = args.augm
print('patch size',patch_size)
if patch_size==0 or augm==0:
    print('error type!')
    import sys
    sys.exit(0)

in_chan = patch_size

data_type = 'agum_'+str(augm)+'_crop_256_patch_'+str(in_chan)
loss_type = args.loss_type
save_path = main_path + 'test_result/save_each_epoch/'+ patient_type+'/'+data_type\
            +'_ws_'+str(args.window_size)+'_'+loss_type+'/'
if not os.path.exists(save_path):
        os.makedirs(save_path)
weigth_input_path = main_path +'/work_dirs/weight/'+data_type+'_save_each_eopch_'+'ws_'+str(args.window_size)\
                    +'_'+loss_type+'/'

with open(save_path+'log_'+data_type+'_ws_'+str(args.window_size)+'_'+'.txt','a') as f:    #设置文件对象
        f.truncate(0)

num_workers = 64
for j in range(len(Patient_list)):
    name_num = j
    max_dice = 0
    best_epoch = 0
    best_index = []
    print(Patient_list[name_num])

    #get best epoch 
    csv_path = save_path+Patient_list[name_num]+'.csv'

    datas = pd.read_csv(csv_path)
    weigth_number = int(datas['Best_dice_epoch'][0])
    best_dice = datas['Dice'][0]

    
    print('Best_dice_epoch',weigth_number)
    print('Best_dice ',best_dice)
    test_path = main_data_path +'test_data/'+data_type+'/'+ Patient_list[name_num]
    test_set = load_data(test_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                batch_size=32,
                                shuffle=False,num_workers=num_workers)

    weigth_path = weigth_input_path + str(weigth_number)+'.pth'

    device = torch.device("cuda")
    model = MCM_Hit(n_channels=in_chan, n_classes=1, window_size = window_size).to(device)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    model.to(device)
    checkpoint = torch.load(weigth_path)  
    model.load_state_dict(checkpoint['model'])
    #model.load_state_dict(torch.load(weigth_path,map_location=device))   
    model.eval()

    data_label = []
    data_pred = []
    data_image = []

    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.float()
            image = image.to(device)
            Output = model(image)

            slice_pred = torch.squeeze(Output,axis = 1)
            data_pred.append(slice_pred)
            

    pred = torch.vstack(data_pred)
    pred = pred.permute(1,2,0)
    index = 0
    for image, label in (test_loader):
        img_shape = image.shape[1]
        if img_shape > 2:
            index = 1
            break;
    #print(index)
    for image, label in (test_loader):
        #print(image.shape)
        if index == 1:
            Sl = int(img_shape/2)
            image = image[:,Sl,:,:]
            image = np.squeeze(image,axis = 1)           
        slice_image = np.squeeze(image,axis = 1)
        slice_masks = np.squeeze(label,axis = 1)
        data_label.append(slice_masks)
        data_image.append(slice_image)

    masks = np.vstack(data_label)
    masks = masks.transpose(1,2,0)
    img = np.vstack(data_image)
    img = img.transpose(1,2,0)
    print('pred shape:{},masks shape:{}'.format(pred.shape,masks.shape))

    pred[pred>=0.5]=1
    pred[pred<0.5]=0

    pred = pred.detach().cpu().numpy()
    index = Index(pred, masks)
    print('Dice',index[0])
    # make include tumor slice
    in_slice = []
    for i in range(masks.shape[-1]):
        if masks[:, :, i].max() == 1:
            in_slice.append(i)
    print('label include tumor slice:', in_slice)
    print('label include tumor slice length:', len(in_slice))

    #record pred include tumor slice
    pred_slice = []
    for i in range(masks.shape[-1]):
        if pred[:,:,i].max() == 1:
            pred_slice.append(i)
    print('pred include tumor slice:',pred_slice)
    print('pred include tumor slice length:',len(pred_slice))
    print('\n')
    
    save_img_label = save_path+'image/'+str(Patient_list[name_num])+'/label/'
    if not os.path.exists(save_img_label):
        os.makedirs(save_img_label)
    save_img_pred = save_path+'image/'+str(Patient_list[name_num])+'/pred/'
    if not os.path.exists(save_img_pred):
        os.makedirs(save_img_pred)
    #list to str
    str_in_slice = ' '.join(str(i) for i in in_slice)
    str_in_slice = 'label include tumor slice:' +str_in_slice
    str_in_slice_len = 'pred include tumor slice length:'+str(len(in_slice))
    str_pred_slice = ' '.join(str(i) for i in pred_slice)
    str_pred_slice = 'pred include tumor slice:'+str_pred_slice
    str_pred_slice_len = 'pred include tumor slice length:'+str(len(pred_slice))
    str_list = []
    str_list.append(Patient_list[name_num])
    str_list.append(str_in_slice)
    str_list.append(str_in_slice_len)
    str_list.append(str_pred_slice)
    str_list.append(str_pred_slice_len)
    #all
    with open(save_path+'log_'+data_type+'_ws_'+str(args.window_size)+'_'+'.txt','a') as f:    #设置文件对象
        f.writelines('%s\n'% s for s in str_list)


    #label
    plt.figure()
    for i in range(len(in_slice)):
        num = in_slice[i]
        x = masks[:,:,num]
        y = img[:,:,num]
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y,cmap='gray')
        plt.title(str(num))
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x,cmap='gray')
        plt.title(str(num))
        plt.suptitle('raw')
        plt.savefig(save_img_label + str(num)+'_label'+'.png')

    #pred
    plt.figure()
    for i in range(len(pred_slice)):
        num = pred_slice[i]
        x = pred[:, :, num]
        y = img[:,:,num]
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y,cmap='gray')
        plt.title(str(num))
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x,cmap='gray')
        plt.title(str(num))
        plt.suptitle('pred')
        plt.savefig(save_img_pred + str(num)+'_pred'+'.png')
f.close()

    