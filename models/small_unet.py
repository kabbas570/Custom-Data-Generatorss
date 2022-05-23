import torch
import torch.nn as nn
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, 3,padding=1,stride=2),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

def double_conv_u(in_channels, out_channels):
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
                
        self.dconv_down1 = double_conv(16, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
     
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        

        self.dconv_up1 = double_conv_u(128 + 256, 128)
        self.dconv_up2 = double_conv_u(128 + 64, 64)
        self.dconv_up3 = double_conv_u(64 + 16, 16)
        
        
        self.conv_1 = nn.Conv2d(1, 16, 1)
        self.conv_last = nn.Conv2d(16, 1, 1)
        self.activation = torch.nn.Sigmoid()
        
        
    def forward(self, x_in):
        x1=self.conv_1(x_in)
        conv1 = self.dconv_down1(x1)      
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        
        u1=self.upsample(conv3)  
        cat1 = torch.cat([u1, conv2], dim=1) 
        u2=self.dconv_up1(cat1)
        
        u3=self.upsample(u2)  
        cat2 = torch.cat([u3, conv1], dim=1) 
        u3=self.dconv_up2(cat2)
        
        u4=self.upsample(u3)  
        cat3 = torch.cat([u4, x1], dim=1) 
        u4=self.dconv_up3(cat3)
        
        out=self.conv_last(u4)
        
       
        
        return out
    
def model() -> UNet_1:
    model = UNet_1()
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
print(model)
model.to(device=DEVICE,dtype=torch.float)
summary(model, (1,48,48))
