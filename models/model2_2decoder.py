import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )   
def last_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding='same')
    ) 
def last_conv1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding='same')
    ) 

def last_conv_sc():
    return nn.Sequential(
        nn.Conv2d(192, 16, 3, padding='same'),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, 1, padding='same'),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, 3, padding='same'),
        nn.ReLU(inplace=True)
    ) 
class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last_seg = last_conv(64,1)
        self.conv_last_b = last_conv(64,1)
        self.conv_last_sc = last_conv1(16,1)
        self.activation = torch.nn.Sigmoid()
        self.last_conv_sc=last_conv_sc()
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x_f = self.dconv_down4(x)
        
        x = self.upsample(x_f)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out1 = self.conv_last_seg(x)
        
        #decoder 2
        x1 = self.upsample(x_f)        
        x1 = torch.cat([x1, conv3], dim=1)
        
        x1 = self.dconv_up3(x1)
        x1 = self.upsample(x1)        
        x1 = torch.cat([x1, conv2], dim=1)       

        x1 = self.dconv_up2(x1)
        x1 = self.upsample(x1)        
        x1= torch.cat([x1, conv1], dim=1)          
        x1 = self.dconv_up1(x1)
        out2 = self.conv_last_b(x1)
        
        out= torch.cat([x ,x1,conv1], dim=1) 
        out=self.last_conv_sc(out)
        out = self.conv_last_sc(out)
        return self.activation(out1),self.activation(out2),self.activation(out)
