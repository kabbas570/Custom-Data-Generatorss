
import torch
from torch import nn
from torch.nn import init



class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual








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

def res_conv(in_channels, out_channels,stride=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )      
class m_unet(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()
        
        self.conv0=conv1_(1,16)
        self.conv1=conv1_(16,32)
        self.conv2 = double_conv(32,64)
        self.conv3 = double_conv(64,128)
        self.conv4 = double_conv(128,256)
        self.conv5 = double_conv(256,512)
        self.conv6 = double_conv(512,1024)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv1_u = double_conv(1024,512)
        self.conv2_u = double_conv(1024,256)
        self.conv3_u = double_conv(512,128)
        self.conv4_u = double_conv(256,64)
        self.conv5_u = double_conv(128,32)
        self.conv6_u = double_conv(64,16)
        self.conv7_u = double_conv(32,16)
    
        self.conv_last =  nn.Conv2d(32, 1, 1)
        self.conv_last1 =  nn.Conv2d(32, 1, 1)
        self.conv_last2 =  nn.Conv2d(64, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.activation = torch.nn.Sigmoid()
        
        ## res paths ###
        
        self.res_path1=res_conv(2048-512,512)
        self.res_path2=res_conv(2048-256,256)
        self.res_path3=res_conv(1024-128,128)
        self.res_path4=res_conv(512-64,64)
        self.res_path5=res_conv(272-32,32)
        self.res_path6=res_conv(128-16,16)
        self.res_path7=res_conv(64-16,16)
        
        
        self.cbam1=CBAMBlock(512)
        self.cbam2=CBAMBlock(256)
        self.cbam3=CBAMBlock(128)
        self.cbam4=CBAMBlock(64)
        self.cbam5=CBAMBlock(32)
        self.cbam6=CBAMBlock(16)
        self.cbam7=CBAMBlock(16)

    def forward(self,x1,x2,x3):
        
        ### Encoder Design ###
           
           
           ## conv1  ###
        conv0_1 = self.conv0(x1)
        conv0_2 = self.conv0(x2)
        conv0_3 = self.conv0(x3)   
    
        ## conv1  ###
        conv1_1 = self.conv1(conv0_1)
        conv1_2 = self.conv1(conv0_2)
        conv1_3 = self.conv1(conv0_3)      
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
        
        ## conv5  ###
        conv6_1 = self.conv6(conv5_1)
        conv6_2 = self.conv6(conv5_2)
        conv6_3 = self.conv6(conv5_3) 
        conv6_1 = self.maxpool(conv6_1)
        conv6_2 = self.maxpool(conv6_2)
        conv6_3 = self.maxpool(conv6_3)
        
        
        ### Decoder Design ###
        
        ### Block 1 ###
        u1 = self.conv1_u(conv6_3)
        u1 = self.upsample(u1)
        c1 = torch.cat([conv5_3,conv6_2], dim=1)
        
        r1=self.res_path1(c1)
        
        r1=self.cbam1(r1)
        u1=torch.cat([u1,r1], dim=1)
        
        # ### Block 2 ###
        u2 = self.conv2_u(u1)
        u2 = self.upsample(u2)
        c2 = torch.cat([conv6_1,conv4_3,conv5_2], dim=1)
        
        r2=self.res_path2(c2)
        
        r2=self.cbam2(r2)
        u2=torch.cat([u2,r2], dim=1)
        
        # ### Block 3 ###
        u3 = self.conv3_u(u2)
        u3 = self.upsample(u3)
        c3 = torch.cat([conv5_1,conv3_3,conv4_2], dim=1)

        r3=self.res_path3(c3)
        
        r3=self.cbam3(r3)
        u3=torch.cat([u3,r3], dim=1)
        
        # ### Block 4 ###
        u4 = self.conv4_u(u3)
        u4 = self.upsample(u4)
        c4 = torch.cat([conv4_1,conv2_3,conv3_2], dim=1)
        
        r4=self.res_path4(c4)
        
        
        r4=self.cbam4(r4)
        
        u4=torch.cat([u4,r4], dim=1)
        
        # ### Block 5 ###
        u5 = self.conv5_u(u4)
        u5 = self.upsample(u5)
        c5 = torch.cat([conv1_3,conv2_2,conv3_1,conv0_3], dim=1)
        
        r5=self.res_path5(c5)
        
        
        r5=self.cbam5(r5)
        u5=torch.cat([u5,r5], dim=1)
        
        # ### Block 6 ###
        u6 = self.conv6_u(u5)
        u6 = self.upsample(u6)
        c6 = torch.cat([conv1_2,conv2_1,conv0_2], dim=1)
        
        r6=self.res_path6(c6)
        
        r6=self.cbam6(r6)
        u6=torch.cat([u6,r6], dim=1)
        
        # ### Block 7 ###
        u7 = self.conv7_u(u6)
        u7 = self.upsample(u7)
        c7 = torch.cat([conv1_1,conv0_1], dim=1)
        
        r7=self.res_path7(c7)
        
        r7=self.cbam7(r7)
        u7=torch.cat([u7,r7], dim=1)
        
        out1=self.conv_last(u7)
        out2=self.conv_last1(u6)
        out3=self.conv_last2(u5)
        
        return self.activation(out1),self.activation(out2),self.activation(out3)  
