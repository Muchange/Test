import torch
import lib
import argparse
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from utils.metrics import CrossEntropyLoss,Diceloss,Focalloss,Train_index
from utils.dataset_augm import *
from models.model import MCM_Hit_mor
import cv2
from functools import partial
from random import randint
import timeit
from thop import profile
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
import time
from utils.getlogger import get_logger
import sys
from torchvision import transforms

torch.cuda.empty_cache( )
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=0.00005, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--learning_decay', type=int, default=25)
parser.add_argument('--loss_type', type=str, default='Dice')
parser.add_argument('--resume', type=str, default=' ')
parser.add_argument('--patch_size', type=int, default=0)
parser.add_argument('--dataset_type', type=str, default=' ')
parser.add_argument('--window_size', type=int, default=0)
parser.add_argument('--drop', type=float, default=0)

args = parser.parse_args()

'''
python train_single_save_each_epoch.py --loss_type CE+Dice_config_0 --resume False --dataset_type nodule --patch_size 3 --window_size 8

'''

Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')
drop_rate  = args.drop
patch_size = args.patch_size
dataset_type = args.dataset_type
if patch_size==0:
    print('error patch size!')
    import sys
    sys.exit(0)

# augm = args.augm
# if augm==0:
#     print('error augm!')
#     import sys
#     sys.exit(0)

window_size = (args.window_size,args.window_size)
if window_size[1]==0:
    print('error window size!')
    import sys
    sys.exit(0)

if args.resume=='True':
    resume = 1
elif args.resume=='False':
    resume = 0
else:
    print('error show type!')
    sys.exit(0)


main_data_path = '/media/xd/date/muzhaoshan/Synapse data/RawData/'
main_path = '/media/xd/date/muzhaoshan/MCM_Hit_Mor/data/'

in_chan = patch_size

data_type = 'crop_256_patch_'+str(in_chan)

#data_type = 'train_data_crop_256'
train_path = main_data_path +'train_data/'+ data_type+'/'
#val_path = main_path + 'train_data/'+data_type+'/val'
loss_type  = args.loss_type
record_type = data_type + '_'+loss_type      #save train loss logger ==> jpg

save_patch = main_path +'/work_dirs/weight/'+dataset_type+'_'+data_type+'_save_each_eopch_'+'ws_'+str(args.window_size)\
            +'_'+loss_type+'/'
if not os.path.exists(save_patch.replace('weight','logger')):
    os.makedirs(save_patch.replace('weight','logger'))
logger = get_logger(save_patch.replace('weight','logger')+str(Time)+'.log')


# train_dataset = load_data(train_path)
train_dataset =  load_data(train_path,'train',
                            transform=transforms.Compose(
                                        [RandomGenerator()]))
#val_dataset = load_data(val_path)
trianloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,
                            num_workers=args.num_workers,drop_last=True)

##=========configs===========##
logger.info('Data Type {}'.format(data_type))
logger.info('max epoch {}'.format(args.epochs))
logger.info('batch size {}'.format(args.batch_size))
logger.info('learning rate {}'.format(args.learning_rate))
logger.info('learning decay {}'.format(args.learning_decay))
logger.info('loss type {}'.format(loss_type))
logger.info('patch size {}'.format(in_chan))
# logger.info('drop rate {}'.format(drop_rate))
logger.info('window size {}'.format(window_size))
logger.info('dataset type {}'.format(dataset_type))
logger.info('resume {}'.format(args.resume))



model = MCM_Hit_mor(n_channels=in_chan, n_classes=1, window_size = window_size)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")
model = nn.DataParallel(model, device_ids=[0]).cuda()
model.to(device)

criterion_ce = CrossEntropyLoss()
criterion_dice = Diceloss()
criterion_fl = Focalloss()


optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_decay, gamma=0.99)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('Total_params: {} M'.format(pytorch_total_params/1000000.0))
log_dict = {'celoss': [],'diceloss': [],'flloss': [],'loss': [],'dice': [],'epoch':[] }

start_epoch = 0
#resume = True
if resume:
    logger.info('==============using checkpoint!================')
    save_patch_number = len(os.listdir(save_patch)) - 1 
    start_epoch = save_patch_number
    path_checkpoint = save_patch+str(start_epoch)+'.pth'  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点

    model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数

    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
mdice_max = 0
logger.info('start training!')
for epoch in range(start_epoch+1,args.epochs+1):
    print('epoch',epoch)
    epoch_running_loss = 0
    train_idx = np.zeros(6)
    step_loss_ce = 0
    step_loss_dice = 0
    step_loss_fl = 0
    step_loss = 0
    trainloader_len = len(trianloader)
    model.train()
    for image, label in tqdm(trianloader):
        print(image.shape,label.shape)
        #print(image.dtype,label.dtype)
        image = image.float()
        label = label.float()
        image, label = image.to(device), label.to(device)
        output = model(image)

        loss_ce = criterion_ce(output, label)
        loss_dice = criterion_dice(output, label)
        # loss_fl = criterion_fl(output, label)
        loss = loss_ce + loss_dice
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ===================loss====================
        step_loss_ce += loss_ce.item()
        step_loss_dice += loss_dice.item()
        # step_loss_fl += loss_fl.item()
        step_loss += loss.item()
        
        # ===================index====================
        train_index = Train_index(label,output)
        #print(train_index)
        train_idx += train_index
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    scheduler.step()
    step_loss_ce /= trainloader_len
    step_loss_dice /= trainloader_len
    step_loss_fl /= trainloader_len
    step_loss /= trainloader_len
    # logger.info('epoch [{}/{}], lr:{:.8f}, CrossEntropyLoss:{:.6f}, Diceloss:{:.6f},Focalloss:{:.6f}, Loss:{:.6f}'
    #             .format(epoch, args.epochs,lr,step_loss_ce,step_loss_dice,step_loss_fl,step_loss))

    logger.info('epoch [{}/{}], lr:{:.8f}, CrossEntropyLoss:{:.6f}, Diceloss:{:.6f},Loss:{:.6f}'
                .format(epoch, args.epochs,lr,step_loss_ce,step_loss_dice,step_loss))

    train_idx /= trainloader_len
    logger.info('mdice:{:.6f}, macc:{:.6f}, mIou:{:.6f}, mF_score:{:.6f}, mPrecision:{:.6f},mRecall:{:.6f}'\
                .format(train_idx[0],train_idx[1], train_idx[2], train_idx[3],train_idx[4],train_idx[5]))

    log_dict['celoss'].append(float(step_loss_ce))
    log_dict['diceloss'].append(float(step_loss_dice))
    log_dict['flloss'].append(float(step_loss_fl))
    log_dict['loss'].append(float(step_loss))
    log_dict['dice'].append(float(train_idx[0]))
    log_dict['epoch'].append(epoch)

    logger.info('save epoch {} model'.format(epoch))
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }
    if not os.path.exists(save_patch):
        os.makedirs(save_patch)
    torch.save(checkpoint, save_patch + str(epoch) + ".pth")

    
plt.figure()
plt.plot(log_dict['epoch'],log_dict['celoss'],\
        log_dict['epoch'],log_dict['diceloss'],\
        log_dict['epoch'],log_dict['flloss'],\
        log_dict['epoch'],log_dict['loss'],\
        log_dict['epoch'],log_dict['dice'])
#plt.plot(log_dict['epoch'],log_dict['loss'],log_dict['epoch'],log_dict['dice'])
label = ['loss_ce','loss_dice','loss','dice']
plt.legend(label,loc='best')
record_path = main_path + 'record/train_save_each_epoch/'
if not os.path.exists(record_path):
    os.makedirs(record_path)
plt.savefig(record_path + record_type+'.jpg')
