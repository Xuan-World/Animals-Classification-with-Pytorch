import os

import torch
from torch import nn
import torchvision
from PIL import Image
import csv
def test(net,input_dir,output_dir,categories):
    net.eval()
    images = []
    for item in os.listdir(input_dir):
        images.append(item)
    print(images)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(224),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                 [0.229, 0.224, 0.225])
                                                ])
    features=[transform( Image.open(os.path.join(input_dir,i)) ) for i in images]
    labels=[]
    for i in features:
        idx=int(net(i.unsqueeze(0)).argmax(axis=1))
        labels.append(categories[idx])
    with open(os.path.join(output_dir,"result.csv"),'w') as f:
        csv_writter=csv.writer(f)
        csv_head=['filename','category']
        csv_writter.writerow(csv_head)
        for i in range(len(labels)):
            csv_writter.writerow([images[i],labels[i]])

