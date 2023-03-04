import torch
from torch import  nn
from draw import drawPlot
import time
class Train():
    def __init__(self,**kwargs):
        self.lr=kwargs['lr']#学习率
        self.wd=kwargs['weight_decay']#权重衰减——定义正则项的系数
        self.gamma=kwargs['scheduler_gamma']#lr会随着训练epoch的迭代而变小。scheduler可以管理lr的衰减。gamma为衰减率。
        self.gpus=[torch.device("cuda",i) for i in kwargs['gpus']]#'gpus':[0]
        self.epochs=kwargs['epochs']#epoch的个数


    def model_acc(self,y,y_hat):
        return (y_hat.argmax(axis=1)==y).float().sum()
    def evaluate(self,net,valid_dataloader,valid_acc,device):
        with torch.no_grad():
            for X,y in valid_dataloader:
                X=X.to(device)
                y=y.to(device)
                y_hat=net(X)
                valid_acc[0]+=self.model_acc(y,y_hat)#batch中判别正确的个数
                valid_acc[1]+=y.numel()#batch中的数据个数
            return valid_acc

    def train(self,net,train_dataloader,valid_dataloader,logger):
        loss=nn.CrossEntropyLoss(reduction='none')
        weight_optim=torch.optim.SGD(net.parameters().weight,lr=self.lr,weight_decay=self.wd)
        bias_optim=torch.optim.SGD(net.parameters().bias,lr=self.lr)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optim,gamma=self.gamma)
        device=self.gpus[0]
        net=net.to(device)
        train_acc_list=[]#为了最终画图用，所以要存下变化情况
        train_loss_list=[]
        valid_acc_list=[]
        for epoch in range(self.epochs):
            net.train()
            train_acc=[0.0]*2
            train_loss=[0.0]*2
            valid_acc=[0.0]*2
            total_time=0
            for X,y in train_dataloader:
                start_time=time.time()
                X=X.to(device)
                y=y.to(device)
                y_hat=net(X)
                l=loss(y_hat,y).sum()
                optim.zero_grad()
                l.backward()
                optim.step()
                with torch.no_grad():
                    train_acc[0]+=self.model_acc(y,y_hat)
                    train_acc[1]+=y.numel()
                    train_loss[0]+=l
                    train_loss[1]+=y.numel()
                total_time += time.time()-start_time

            net.eval()
            valid_acc=self.evaluate(net,valid_dataloader,valid_acc,device)
            logger.info(f"epoch{epoch}:train_loss:{train_loss[0]/train_loss[1]:.4f},"
                  f"train_acc:{train_acc[0]}/{train_acc[1]}={train_acc[0]/train_acc[1]:.4f},valid_acc:{valid_acc[0]}/{valid_acc[1]}={valid_acc[0]/valid_acc[1]:.4f},speed={train_loss[1]/total_time:.1f}items/sec")
            train_loss_list.append(train_loss[0]/train_loss[1])
            train_acc_list.append(train_acc[0]/train_acc[1])
            valid_acc_list.append(valid_acc[0]/valid_acc[1])
            scheduler.step()
        drawPlot(train_loss_list,self.epochs,"epochs","train_loss","训练集上损失的迭代变化")
        drawPlot(train_acc_list,self.epochs,"epochs","train_acc","训练集上精度的迭代变化")
        drawPlot(valid_acc_list,self.epochs,"epochs","valid_acc","测试集上精度的迭代变化")
