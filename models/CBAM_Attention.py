import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
           nn.Flatten(),
           nn.Linear(channel, channel // reduction),
           nn.ReLU(),
           nn.Linear(channel // reduction, channel)
           )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.mlp(max_result)
        avg_out=self.mlp(avg_result)
        output=max_out+avg_out
        
        scale = self.sigmoid(output).unsqueeze(2).unsqueeze(3).expand_as(x)
        print(scale.shape)
        return scale



class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=1):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding='same')
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock1(nn.Module):

    def __init__(self, channel,reduction=16):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention()

    def forward(self, x):
        
        residual=x
        out_ca=self.ca(x)
        refined=residual*out_ca
        
        out_sa=self.sa(refined)
        return out_sa
