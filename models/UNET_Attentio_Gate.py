import torch
from torch import nn

class Channel_1(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()

            
        self.conv1_x=nn.Sequential(
            nn.Conv2d(in_channels, out_channels,1,bias=False,stride=2)
        )
        
        self.conv1_g=nn.Sequential(
            nn.Conv2d(out_channels, out_channels,1,bias=False)
        )
        
        self.activation1=nn.ReLU()
        
        self.conv1_1=nn.Sequential(
            nn.Conv2d(out_channels,1,1,bias=False)
        )
        
        self.activation2 = torch.nn.Sigmoid()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
    
    def forward(self, x,g) :
        
        x_conv1=self.conv1_x(x)
        g_conv1=self.conv1_g(g)
        
        
        sum_=x_conv1+g_conv1
        
        sum_ReLU=self.activation1(sum_)
        
        att_f=self.conv1_1(sum_ReLU)
        
        att_f=self.activation2(att_f)
        
        out = self.upsample(att_f)
        
        return out
        
        
class Attention_Block(nn.Module):

    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.A1=Channel_1(in_channels=in_channels, out_channels=out_channels)

    def forward(self,x,g):
        residual1=x

        out=self.A1(x,g)
        out1=out*residual1

        return out1

def double_conv0(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
    )

def double_conv3(in_channels, out_channels,f_size,p_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
        
    ) 

def double_conv1(in_channels, out_channels,f_size,p_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

def double_conv_u(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ) 

class UNet_2(nn.Module):

    def __init__(self, input_channels=1,n_class=1):
        super().__init__()
                
        self.dconv_down1 = double_conv0(1, 64,7)
        self.dconv_down2 = double_conv1(64, 128,7,3)
        self.dconv_down3 = double_conv1(128, 256,5,2)
        self.dconv_down4 = double_conv1(256, 512,3,1)
        self.dconv_down5 = double_conv3(512,1024,3,1)
       
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)


        self.dconv_up1 = double_conv_u(512 + 512, 512,3)
        self.dconv_up2 = double_conv_u(256 + 256, 256,3)
        self.dconv_up3 = double_conv_u(128+128, 128,5)
        self.dconv_up4 = double_conv_u(64+64,64,5)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        self.activation = torch.nn.Sigmoid()
        
        
        ### attention gates ####
        
        self.att_1=Attention_Block(512,1024)
        self.att_2=Attention_Block(256,512)
        self.att_3=Attention_Block(128,256)
        self.att_4=Attention_Block(64,128)
        
        
    def forward(self, x_in):
        conv1 = self.dconv_down1(x_in)      
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)
        
        
        ## decoder ####
        
        g=conv5
        x=conv4
        att1= self.att_1(x,g)
        u1=self.up1(conv5)
        u1 = torch.cat([u1, att1], dim=1) 
        u1=self.dconv_up1(u1)
        
        g=u1
        x=conv3
        

        att2= self.att_2(x,g)
        
        u2=self.up2(u1)
        u2 = torch.cat([u2, att2], dim=1) 
        u2=self.dconv_up2(u2)
        
        g=u2
        x=conv2
        

        
        att3= self.att_3(x,g)
        
        u3=self.up3(u2)
        u3 = torch.cat([u3, att3], dim=1) 
        u3=self.dconv_up3(u3)
        
        g=u3
        x=conv1
            
        att4= self.att_4(x,g)
        
        u4=self.up4(u3)
        u4 = torch.cat([u4, att4], dim=1) 
        u4=self.dconv_up4(u4)
        
        out=self.conv_last(u4)
        
        return self.activation(out)
