import os.path

import torch
import torchvision.datasets
from torch.utils import data
from PIL import Image
from experiments.check_error import checkErrorImgs
class MyDataset():
    def __init__(self,data_dir,train_batch_size,val_batch_size,num_workers,if_check,*args,**kwargs):
        self.data_dir=data_dir
        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size
        self.test_batch_size=val_batch_size
        self.num_workers=num_workers
        if if_check:
            print("checking all images")
            checker=checkErrorImgs(os.path.join(self.data_dir, "Training Data"))
            checker.doDelete()
            checker = checkErrorImgs(os.path.join(self.data_dir, "Validation Data"))
            checker.doDelete()
            checker = checkErrorImgs(os.path.join(self.data_dir, "Testing Data"))
            checker.doDelete()
        else:
            print("skip the check of all images")

    def train_dataloader(self):
        path = os.path.join(self.data_dir, "Training Data")
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                                       saturation=0.2),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])])
        dataset=torchvision.datasets.ImageFolder(path,transform)
        dataloader = data.DataLoader(dataset=dataset, batch_size=self.train_batch_size, shuffle=True,
                                          num_workers=self.num_workers)
        return dataloader
    def valid_dataloader(self):
        path = os.path.join(self.data_dir, "Validation Data")
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(224),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])
                                            ])
        dataset=torchvision.datasets.ImageFolder(path,transform)
        dataloader = data.DataLoader(dataset=dataset, batch_size=self.val_batch_size, shuffle=False,
                                     num_workers=self.num_workers)
        return dataloader
    def test_dataloader(self):
        path = os.path.join(self.data_dir, "Testing Data")
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                            torchvision.transforms.CenterCrop(224),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])
                                            ])
        dataset=torchvision.datasets.ImageFolder(path,transform)
        dataloader = data.DataLoader(dataset=dataset, batch_size=self.test_batch_size, shuffle=False,
                                     num_workers=self.num_workers)
        return dataloader
