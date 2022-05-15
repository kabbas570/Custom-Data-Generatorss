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
    
class UNet_1(nn.Module):

    def __init__(self, input_channels=1,n_class=1):
        super().__init__()
                
        self.dconv_down1 = double_conv(input_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.dconv_up3c = double_conv(384, 256)
        self.dconv_up2c = double_conv(384, 128)
        self.dconv_up1c = double_conv(192, 64)
        
        self.conv_last1 = nn.Conv2d(64, 1, 1)
        self.conv_last2 = nn.Conv2d(64, 1, 1)
        self.activation = torch.nn.Sigmoid()
        
        
    def forward(self, x_in):
        conv1 = self.dconv_down1(x_in)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last1(x)
        
        ### Scar Decoder ###
        
        conv1c = self.dconv_down1(x_in)
        s1 = self.maxpool(conv1c)

        conv2c = self.dconv_down2(s1)
        s2 = self.maxpool(conv2c)
        conv3c = self.dconv_down3(s2)
        s2 = torch.cat([s2, conv3c], dim=1)        
        s3 = self.dconv_up3c(s2)
        s3 = self.upsample(s3)        
        s3 = torch.cat([s3, conv2c], dim=1) 
        s4 = self.dconv_up2c(s3)
        s4 = self.upsample(s4)        
        s4 = torch.cat([s4, conv1c], dim=1)   
        s4 = self.dconv_up1c(s4)
        out1 = self.conv_last2(s4)
        
        return self.activation(out),self.activation(out1)
    
def model() -> UNet_1:
    model = UNet_1()
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, (1, 512, 512))
