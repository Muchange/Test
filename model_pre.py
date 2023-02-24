from itertools import zip_longest
from xml.etree.ElementPath import prepare_descendant
import torch
import lib
import argparse
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from utils.metrics_sy import *
from torch.nn.modules.loss import CrossEntropyLoss
from utils.dataset_synapse_overlap import *
# from models.model import MCM_Hit_mor
from models.model_davit_cgb import MCM_Hit_mor
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
from thop import profile
from test_in_train import testing_in_train
import pandas as pd

torch.cuda.empty_cache()


main_path = '/media/xd/date/muzhaoshan/MCM_Hit_Mor/data//work_dirs/weight/'+\
        'Synapse_crop_256_patch_1_save_each_eopch_CE+Dice_config_cross_2_conv_2_davit/5.pth'
checkpoint = torch.load(main_path) 
ch = checkpoint['model']
# pre_v = np.array(0)
for k,v in ch.items():
    # if 'backbone.main_blocks.1.0.0' in k or 'backbone.main_blocks.1.0.1' in k or 'backbone.norm1' in k:
    # if 'CGB1.cgb.LN' in k:
        print(k)


# #================================model per====================================
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# device = torch.device("cuda")
# model = MCM_Hit_mor(n_channels=1, n_classes=9 , window_size = (8,8)).to(device)
# model = nn.DataParallel(model, device_ids=[0]).cuda()
# print(model)
# #================================model pretrained====================================
# model_dict = model.state_dict()
# # for k,v in model_dict.items():
# #     if 'module.Low.davit.block_spa' in k or 'module.Low.davit.block_chan' in k\
# #         or 'module.Low.davit.low_norm' in k:
# #         print(k)
# model_dict_spa = {k: v for k, v in model_dict.items() if 'module.Low.davit.block_spa' in k}
# model_dict_chan = {k: v for k, v in model_dict.items() if 'module.Low.davit.block_chan' in k\
#         or 'module.Low.davit.low_norm' in k }

# checkpoint = torch.load('Davit_B_UPerNet.pth') 
# ch = checkpoint['state_dict']
# # pre_v = np.array(0)
# # for k,v in ch.items():
# #     # if 'backbone.main_blocks.1.0.0' in k or 'backbone.main_blocks.1.0.1' in k or 'backbone.norm1' in k:
# #     if 'backbone.norm1' in k:
# #         print(k)
# #         pre_v = v

# pretrained_dict_spa = {k: v for k, v in ch.items() if 'backbone.main_blocks.1.0.0' in k}
# pretrained_dict_chan = {k: v for k, v in ch.items() if 'backbone.main_blocks.1.0.1' in k or 'backbone.norm1' in k}


# for m_key,p_key in zip_longest(model_dict_spa.keys(),pretrained_dict_spa.keys()):
#     model_dict_spa[m_key] = pretrained_dict_spa[p_key]

# for m_key,p_key in zip_longest(model_dict_chan.keys(),pretrained_dict_chan.keys()):
#     model_dict_chan[m_key] = pretrained_dict_chan[p_key]

# for k,v in model_dict.items():
#     if 'module.Low.davit.block_spa' in k :
#         model_dict[k] = model_dict_spa[k]
#     if 'module.Low.davit.block_chan' in k or 'module.Low.davit.low_norm' in k:
#         model_dict[k] = model_dict_chan[k]

# model.load_state_dict(model_dict)

# for k,v in model_dict.items():
#     if 'davit.low_norm' in k:
#         print(k)
#         m_v = v
# print((pre_v==m_v).all())
# for k,v in model_dict.items():
#     print(k)

 