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
from utils.load_dataset import load_data
from models.model_ import MCM_Hit
from models.model import MCM_Hit_conv
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

torch.cuda.empty_cache( )

model = MCM_Hit_conv()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")
model = nn.DataParallel(model, device_ids=[0]).cuda()
model.to(device)

criterion_ce = CrossEntropyLoss()
criterion_dice = Diceloss()
criterion_fl = Focalloss()


optimizer = torch.optim.Adam(list(model.parameters()), lr=0.00005,
                             weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99,last_epoch=-1)


start_epoch = 0

for epoch in range(start_epoch+1,5):
    model.train()
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])