import torch
import torch.nn as nn
from typing import Any


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def double_conv(in_channels, out_channels,stride=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3,padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3,padding=1,stride=stride),
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
                
        self.conv2 = double_conv(16,32,stride=8)
        self.conv3 = double_conv(32,64,stride=4)
        self.conv4 = double_conv(64,128,stride=2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        
        self.up_conv1 = up_conv_(128+64,64)
        self.up_conv2 = up_conv_(64+32+64,32)
        self.up_conv3 = up_conv_(32+32,16)
        self.up_conv4 = up_conv_(16+32,16)
        self.up_conv4_o = nn.Conv2d(16,1, 1)
        self.up_conv5 = up_conv_(16+16,16)
        self.up_conv5_o = nn.Conv2d(16,1, 1)
        self.up_conv6 = up_conv_(32,16)
        self.up_conv6_o = nn.Conv2d(16,1, 1)
        self.activation=torch.nn.Sigmoid()
    def forward(self, x1:torch.Tensor,x2:torch.Tensor,x3:torch.Tensor)-> torch.Tensor:
        
        
        ### Encoder Design ###
        
        conv1_1 = self.conv1(x1)
        conv1_2 = self.conv1(x2)
        conv1_3 = self.conv1(x3)
        
        conv2_1 = self.conv2(conv1_1)
        conv2_2 = self.conv2(conv1_2)
        conv2_3 = self.conv2(conv1_3)
        
        conv3_1 = self.conv3(conv2_1)
        conv3_2 = self.conv3(conv2_2)
        
        conv4_1 = self.conv4(conv3_1)
        
        ### Decoder Design ###
        cat1 = torch.cat([conv4_1, conv3_2], dim=1) 
        
        u1 = self.upsample(cat1) 
        u1 = self.up_conv1(u1)
        
        cat2 = torch.cat([u1, conv2_3,conv3_1], dim=1) 
        u2 = self.upsample(cat2) 
        u2 = self.up_conv2(u2)
        
        cat3 = torch.cat([u2,conv2_2], dim=1) 
        u3 = self.upsample(cat3) 
        u3 = self.up_conv3(u3)
        
        cat4 = torch.cat([u3,conv2_1], dim=1) 
        u4 = self.upsample(cat4) 
        u4 = self.up_conv4(u4)
        u4_o=self.up_conv4_o(u4)
        
        cat5 = torch.cat([u4,conv1_3], dim=1) 
        u5 = self.upsample(cat5) 
        u5 = self.up_conv5(u5)
        u5_o=self.up_conv5_o(u5)
        
        cat6 = torch.cat([u5,conv1_2], dim=1) 
        u6 = self.upsample(cat6) 
        print(u6.shape)
        u6 = self.up_conv6(u6)
        u6_o=self.up_conv6_o(u6)
        return self.activation(u6_o),self.activation(u5_o),self.activation(u4_o)
    


