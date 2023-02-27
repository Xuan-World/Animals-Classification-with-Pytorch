import argparse
import torch
import experiments
from experiments import *
import models.resnet
import yaml
import logging
import os

# 读取配置文件
from experiments import dataloader, train

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="filename",
                    help="path to the config file",
                    default="configs/config.yaml")
args = parser.parse_args()
with open(args.filename, 'r') as file:
    configfile = yaml.safe_load(file)
print(configfile)
# 建模型
model = models.resnet.Resnet(**configfile['model_params'])
# 设置logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 设置打印级别
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
if configfile['log']['new_log'] == True and os.path.isfile(configfile['log']['log_dir']):
    os.remove(configfile['log']['log_dir'])
fh = logging.FileHandler(configfile['log']['log_dir'], encoding='utf8')
fh.setFormatter(formatter)
logger.addHandler(fh)

# 加载数据
myData = dataloader.MyDataset(**configfile['data_params'])
train_dataloader = myData.train_dataloader()
valid_dataloader = myData.valid_dataloader()
# 训练开始
trainer = train.Train(**configfile['train_params'])
trainer.train(model, train_dataloader, valid_dataloader, logger)
#保存模型
torch.save(model.state_dict(),"model_static_dict.pth")
