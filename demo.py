import argparse
import yaml
import torch
from experiments import test
from torch import nn
from models import resnet
import os

from PIL import Image
parser=argparse.ArgumentParser()
parser.add_argument('--config','-c',
                  dest="filename",
                  help="input the path of the config file",
                  default="./configs/config.yaml")
args=parser.parse_args()
with open(args.filename,'r') as file:
    configfile=yaml.safe_load(file)

#加载已训练模型
model=resnet.Resnet(**configfile['model_params'])
model.load_state_dict(torch.load('model_static_dict.pth'))


#随机给照片，让其分类,把照片放入demo/input中，我们从output中输出对应照片的分类结果


test.test(model,**configfile['test_params'])



