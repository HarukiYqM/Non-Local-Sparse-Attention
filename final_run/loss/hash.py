from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class HASH(nn.Module):
    def __init__(self):
        super(HASH, self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self, sr, qk, orders, hr, m=3):
        #hash loss
        qk = F.normalize(qk, p=2, dim=1, eps=5e-5)
        N,C,H,W = qk.shape
        qk = qk.view(N,C,H*W)
        qk_t = qk.permute(0,2,1).contiguous()
        similarity_map = F.relu(torch.matmul(qk_t, qk),inplace=True) #[N,H*W,H*W]
        
        orders = orders.unsqueeze(2).expand_as(similarity_map)#[N,H*W,H*W]
        orders_t = torch.transpose(orders,1,2)
        dist = torch.pow(orders-orders_t,2)
        
        ls = torch.mean(similarity_map*torch.log(torch.exp(dist+m)+1))
        ld = torch.mean((1-similarity_map)*torch.log(torch.exp(-dist+m)+1))
        loss = 0.005*(ls+ld)+self.l1(sr,hr) 

        return loss
