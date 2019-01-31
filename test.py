import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
class None_local(nn.Module):
    def __init__(self,in_featrues):
        self.W_theta = nn.Conv3d(in_featrues,in_featrues//2,kernel_size=1)
        self.W_phi = nn.Conv3d(in_featrues,in_featrues//2,kernel_size=1)
        self.W_g = nn.Conv3d(in_featrues,in_featrues//2,kernel_size=1)
        self.W_z = nn.Conv3d(in_featrues//2,in_featrues,kernel_size=1)
        self.outBN = nn.BatchNorm3d(in_featrues,eps = 1e-5,affine=True)
    def forward(self,X):
        Xpooled = F.max_pool3d(kernal_size = (1,2,2),stride = (1,2,2))
        Xt = self.W_theta(Xpooled)
        Xt = F.relu(Xt)
        batch_size = Xpooled.size(0)
        channels = Xpooled.size(1)
        Xt = Xt.view(batch_size,channels//2,-1)
        Xt = Xt.permute(0,2,1)
        Xp = self.W_phi(X)
        Xp = F.relu(Xp)
        Xp = Xp.view(batch_size,channels//2,-1)
        embeded = torch.matmul(Xt,Xp)
        embeded_div = F.softmax(embeded,dim = -1)
        Xg = self.W_g(X)
        Xg = F.relu(Xg).view(batch_size,channels//2,-1).permute(0,2,1)
        Y1 = torch.matmul(embeded_div,Xg)
        Y1 = Y1.view(batch_size,channels//2,Xpooled.size()[2:])
        Y2 = self.W_z(Y1)
        Y3 = F.upsample(self.outBN(Y2),2)
        return X+Y3






        
