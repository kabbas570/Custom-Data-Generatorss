import torch
import torch.nn as nn
from typing import Any


def double_conv(in_channels, out_channels,stride=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

def conv1_(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )   

def up_conv_(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

class m_unet(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()
        
        
        self.conv1=conv1_(1,16)
        self.conv2 = double_conv(16,64)
        self.conv3 = double_conv(64,128)
        self.conv4 = double_conv(128,256)
        self.conv5 = double_conv(256,512)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.activation = torch.nn.Sigmoid()
        
    #def forward(self, x1:torch.Tensor,x2:torch.Tensor,x3:torch.Tensor)-> torch.Tensor:
    def forward(self,x1,x2,x3):
        
        ### Encoder Design ###
        
        
        ## conv1  ###
        conv1_1 = self.conv1(x1)
        conv1_2 = self.conv1(x2)
        conv1_3 = self.conv1(x3)      
        ## conv2  ###
        conv2_1 = self.conv2(conv1_1)
        conv2_2 = self.conv2(conv1_2)
        conv2_3 = self.conv2(conv1_3)      
        conv2_1 = self.maxpool(conv2_1)
        conv2_2 = self.maxpool(conv2_2)
        conv2_3 = self.maxpool(conv2_3)   
        ## conv3  ###
        conv3_1 = self.conv3(conv2_1)
        conv3_2 = self.conv3(conv2_2)
        conv3_3 = self.conv3(conv2_3)   
        conv3_1 = self.maxpool(conv3_1)
        conv3_2 = self.maxpool(conv3_2)
        conv3_3 = self.maxpool(conv3_3)    
        ## conv4  ###
        conv4_1 = self.conv4(conv3_1)
        conv4_2 = self.conv4(conv3_2)
        conv4_3 = self.conv4(conv3_3)  
        conv4_1 = self.maxpool(conv4_1)
        conv4_2 = self.maxpool(conv4_2)
        conv4_3 = self.maxpool(conv4_3)
        
        ## conv5  ###
        conv5_1 = self.conv5(conv4_1)
        conv5_2 = self.conv5(conv4_2)
        conv5_3 = self.conv5(conv4_3) 
        conv5_1 = self.maxpool(conv5_1)
        conv5_2 = self.maxpool(conv5_2)
        conv5_3 = self.maxpool(conv5_3)
        
        print(conv5_1.shape)
        print(conv5_2.shape)
        print(conv5_3.shape)
        
        return conv5_1,conv5_2,conv5_3
