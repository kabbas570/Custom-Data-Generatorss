import torch
import torch.nn as nn

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


class m_unet(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()
        
        
        self.conv1=conv1_(1,16)
        self.conv2 = double_conv(16,64)
        self.conv3 = double_conv(64,128)
        self.conv4 = double_conv(128,256)
        self.conv5 = double_conv(256,512)
        self.conv6 = double_conv(512,1024)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv1_u = double_conv(1024,512)
        self.conv2_u = double_conv(1024,256)
        self.conv3_u = double_conv(512,128)
        self.conv4_u = double_conv(256,64)
        self.conv5_u = double_conv(128,16)
        self.conv6_u = double_conv(112,16)
        
        self.conv_last =  nn.Conv2d(32, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.activation = torch.nn.Sigmoid()
        
    def forward(self,x1):
        
        ### Encoder Design ###
        
        
    
        ## conv1  ###
        conv1_1 = self.conv1(x1)
    
        ## conv2  ###
        conv2_1 = self.conv2(conv1_1)
        conv2_1 = self.maxpool(conv2_1)
        
        ## conv3  ###
        conv3_1 = self.conv3(conv2_1)
 
        conv3_1 = self.maxpool(conv3_1)
        
        ## conv4  ###
        conv4_1 = self.conv4(conv3_1)
        
        conv4_1 = self.maxpool(conv4_1)
       
        
        ## conv5  ###
        conv5_1 = self.conv5(conv4_1)
        conv5_1 = self.maxpool(conv5_1)
       
        ## conv6  ###
        conv6_1 = self.conv6(conv5_1)
        conv6_1 = self.maxpool(conv6_1)
        
        ### Decoder Design ###
        
        # ### Block 1 ###
        u1 = self.conv1_u(conv6_1)
        u1 = self.upsample(u1)
        u1 = torch.cat([u1, conv5_1], dim=1)
        
        # ### Block 2 ###
        u2 = self.conv2_u(u1)
        u2 = self.upsample(u2)
        u2 = torch.cat([u2,conv4_1], dim=1)
        
        # ### Block 3 ###
        u3 = self.conv3_u(u2)
        u3 = self.upsample(u3)
        u3 = torch.cat([u3,conv3_1], dim=1)
        
        # ### Block 4 ###
        u4 = self.conv4_u(u3)
        u4 = self.upsample(u4)
        u4 = torch.cat([u4, conv2_1], dim=1)
        
        # ### Block 5 ###
        u5 = self.conv5_u(u4)
        u5 = self.upsample(u5)
        u5 = torch.cat([u5, conv1_1], dim=1)
        

        
        out=self.conv_last(u5)
        
        
        return self.activation(out)


def model() -> m_unet:
    model = m_unet()
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, (1, 640,640))
#summary(model, [(1, 640,640), (1, 320,320),(1,160,160)])
