import torch
from torch import nn
import torchvision
from torchsummary import summary

class Resnet(nn.Module):
    def __init__(self, layers_num, pretrained, classes_num):
        super().__init__()
        self.net=nn.Sequential()
        if layers_num == 50:
            self.net.resnet=torchvision.models.resnet50(pretrained=pretrained)
        elif layers_num == 18:
            self.net.resnet=torchvision.models.resnet18(pretrained=pretrained)
        elif layers_num == 34:
            self.net.resnet=torchvision.models.resnet34(pretrained=pretrained)
        else:
            print("暂不支持层数，从18,34,50中挑选")
            return

        self.net.outputLayer=nn.Sequential(nn.Linear(1000,256),nn.ReLU(),nn.Linear(256,classes_num))

        for param in self.net.resnet.parameters():
            param.requires_grad = False
        for layer in self.net.outputLayer:
            if type(layer) ==nn.Linear:
                nn.init.xavier_uniform_(layer.weight)


    def forward(self, X):
        return self.net(X)

net=Resnet(18,True,10).to("cuda")
summary(net,(3,224,224))