import numpy as np
from torch import nn
import torch
import math

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class CNNEncoder(nn.Module):
    def __init__(self, chnum_in):
        super(CNNEncoder, self).__init__()
        self.chnum_in = chnum_in
        fea_num_x1 = 4
        fea_num_x2 = 8
        fea_num_x3 = 16
        fea_num_x4 = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.chnum_in, fea_num_x1, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(fea_num_x1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fea_num_x1, fea_num_x2, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(fea_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fea_num_x2, fea_num_x3, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(fea_num_x3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fea_num_x3, fea_num_x4, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(fea_num_x4),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        output = l2_norm(x)
        return output

class Arcface(nn.Module):
    def __init__(self,embedding_size=128,classnum=6,s=64.,m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.randn(embedding_size,classnum))
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # 0.2397127693021015
        self.threshold = math.cos(math.pi - m)  # -0.8775825618903726
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)  # (B,classnum)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)  # (B,classnum)
        # cond_v = cos_theta - self.threshold
        # cond_mask = (cond_v <= 0)
        # keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        # cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)  # [0,1,...,nB-1]
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))