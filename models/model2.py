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

class M_UNet(nn.Module):

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
        self.up_conv4_o = up_conv_(16,1)
        self.up_conv5 = up_conv_(16+16,16)
        self.up_conv5_o = up_conv_(16,1)
        self.up_conv6 = up_conv_(16+16,1)
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
        u6 = self.up_conv6(u6)
        return self.activation(u6),self.activation(u5_o),self.activation(u4_o)
    
def m_unet(**kwargs: Any) -> M_UNet:
    model = M_UNet()
    return model


from torchsummary import summary
model = m_unet()
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(1, 512, 512),(1, 256, 256),(1, 128, 128),])


batch_size=1
image_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/img'
mask_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/seg_gt/'

from g4 import Data_Loader
val_loader=Data_Loader(image_path,mask_path,batch_size)

from tqdm import tqdm
loop = tqdm(val_loader)
for batch_idx, (img1,img2,img3,gt,label) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            img2 = img2.to(device=DEVICE,dtype=torch.float)
            img3 = img3.to(device=DEVICE,dtype=torch.float)
            
            p1=model(img1,img2,img3)


