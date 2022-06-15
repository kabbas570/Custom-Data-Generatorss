
import torch
from torch import nn


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
        
        self.conv0=  double_conv(1,16)
        self.conv1=  double_conv(16,32)
        self.conv2 = double_conv(32,64)
        self.conv3 = double_conv(64,128)
        self.conv4 = double_conv(128,256)
        self.conv5 = double_conv(256,512)
        self.conv6 = double_conv(512,1024)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv1_u = double_conv(1024,512)
        self.conv2_u = double_conv(1024,512)
        self.conv3_u = double_conv(1024,512)
        self.conv4_u = double_conv(512,256)
        self.conv5_u = double_conv(256,128)
        self.conv6_u = double_conv(128,64)
        self.conv7_u = double_conv(64,32)
    
        self.conv_last =  nn.Conv2d(32, 1, 1)
        self.conv_last1 =  nn.Conv2d(64, 1, 1)
        self.conv_last2 =  nn.Conv2d(128, 1, 1)
        
        self.conv_last_SC =  nn.Conv2d(32, 1, 1)
        self.conv_last1_SC =  nn.Conv2d(64, 1, 1)
        self.conv_last2_SC =  nn.Conv2d(128, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.activation = torch.nn.Sigmoid()
        
        ## res paths ###

    def forward(self,x1,x2,x3):
        
        ### Encoder Design ###
           
           
           ## conv0  ###
        conv0_1 = self.conv0(x1)
        conv0_2 = self.conv0(x2)
        conv0_3 = self.conv0(x3)   
        
        #conv0_1,conv0_2,conv0_3= self.att_1(conv0_1,conv0_2,conv0_3)
    
        ## conv1  ###
        conv1_1 = self.conv1(conv0_1)
        conv1_2 = self.conv1(conv0_2)
        conv1_3 = self.conv1(conv0_3)   
        
        #conv1_1,conv1_2,conv1_3= self.att_2( conv1_1,conv1_2,conv1_3)
        
        ## conv2  ###
        conv2_1 = self.conv2(conv1_1)
        conv2_2 = self.conv2(conv1_2)
        conv2_3 = self.conv2(conv1_3)      
        conv2_1 = self.maxpool(conv2_1)
        conv2_2 = self.maxpool(conv2_2)
        conv2_3 = self.maxpool(conv2_3)   
        
        #conv2_1,conv2_2,conv2_3= self.att_3(conv2_1,conv2_2,conv2_3)
        
        ## conv3  ###
        conv3_1 = self.conv3(conv2_1)
        conv3_2 = self.conv3(conv2_2)
        conv3_3 = self.conv3(conv2_3)   
        conv3_1 = self.maxpool(conv3_1)
        conv3_2 = self.maxpool(conv3_2)
        conv3_3 = self.maxpool(conv3_3)   
        
        #conv3_1,conv3_2,conv3_3= self.att_4(conv3_1,conv3_2,conv3_3)
        
        ## conv4  ###
        conv4_1 = self.conv4(conv3_1)
        conv4_2 = self.conv4(conv3_2)
        conv4_3 = self.conv4(conv3_3)  
        conv4_1 = self.maxpool(conv4_1)
        conv4_2 = self.maxpool(conv4_2)
        conv4_3 = self.maxpool(conv4_3)
        
        #conv4_1,conv4_2,conv4_3= self.att_5(conv4_1,conv4_2,conv4_3)
        
        
        ## conv5  ###
        conv5_1 = self.conv5(conv4_1)
        conv5_2 = self.conv5(conv4_2)
        conv5_3 = self.conv5(conv4_3) 
        conv5_1 = self.maxpool(conv5_1)
        conv5_2 = self.maxpool(conv5_2)
        conv5_3 = self.maxpool(conv5_3)
        
        #conv5_1,conv5_2,conv5_3= self.att_6(conv5_1,conv5_2,conv5_3)
        
        ## conv6  ###
        conv6_1 = self.conv6(conv5_1)
        conv6_2 = self.conv6(conv5_2)
        conv6_3 = self.conv6(conv5_3) 
        conv6_1 = self.maxpool(conv6_1)
        conv6_2 = self.maxpool(conv6_2)
        conv6_3 = self.maxpool(conv6_3)
        
        #conv6_1,conv6_2,conv6_3= self.att_7(conv6_1,conv6_2,conv6_3)
        
        
       #        ### Decoder Design ###
#
        batch=conv6_1.shape[0]

        ### Block 1 ###
        u1 = self.conv1_u(conv6_3)
        u1 = self.upsample(u1)
    
        zero_1=torch.zeros([batch,512,10,10],dtype=x1.dtype, device=x1.device, requires_grad=True)
        u1 = torch.cat([u1,zero_1], dim=1)
        conv5_3=torch.cat([conv5_3,zero_1], dim=1)

        u1 = u1 + conv5_3 + conv6_2
        
        
        # # ### Block 2 ###
        u2 = self.conv2_u(u1)
        u2 = self.upsample(u2)
        
        
        zero_2_1=torch.zeros([batch,768,20,20],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_2_2=torch.zeros([batch,512,20,20],dtype=x1.dtype, device=x1.device, requires_grad=True)
        

        
        u2 = torch.cat([u2,zero_2_2], dim=1)
        conv4_3 = torch.cat([conv4_3,zero_2_1], dim=1)
        conv5_2 = torch.cat([conv5_2,zero_2_2], dim=1)
        
        
        u2 = u2+ conv6_1+conv4_3+conv5_2
        
        # # ### Block 3 ###
        u3 = self.conv3_u(u2)
        u3 = self.upsample(u3)
        
        zero_3_1=torch.zeros([batch,384,40,40],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_3_2=torch.zeros([batch,256,40,40],dtype=x1.dtype, device=x1.device, requires_grad=True)
        
        
        #u3 = torch.cat([u3,zero_3_2], dim=1)
        conv3_3 = torch.cat([conv3_3,zero_3_1], dim=1)
        conv4_2 = torch.cat([conv4_2,zero_3_2], dim=1)
        

         
        u3 = u3+ conv5_1+conv3_3+conv4_2

        
        # # ### Block 4 ###
        u4 = self.conv4_u(u3)
        u4 = self.upsample(u4)
        
        zero_4_1=torch.zeros([batch,192,80,80],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_4_2=torch.zeros([batch,128,80,80],dtype=x1.dtype, device=x1.device, requires_grad=True)
        
        
        
        #u4= torch.cat([u4,zero_4_1], dim=1)
        conv2_3 = torch.cat([conv2_3,zero_4_1], dim=1)
        conv3_2 = torch.cat([conv3_2,zero_4_2], dim=1)

        
        u4 =  u4+ conv4_1+conv2_3+conv3_2
        
        # # ### Block 5 ###
        u5 = self.conv5_u(u4)
        u5 = self.upsample(u5)
        
        
        zero_5_1=torch.zeros([batch,96,160,160],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_5_2=torch.zeros([batch,64,160,160],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_5_3=torch.zeros([batch,112,160,160],dtype=x1.dtype, device=x1.device, requires_grad=True)

        
        
        #u5= torch.cat([u5,zero_5_1], dim=1)
        conv1_3= torch.cat([conv1_3,zero_5_1], dim=1)
        conv2_2= torch.cat([conv2_2,zero_5_2], dim=1)
        conv0_3= torch.cat([conv0_3,zero_5_3], dim=1)
                
        
        
        
        u5 = u5+conv1_3+conv2_2+conv3_1+conv0_3
        
        # # ### Block 6 ###
        u6 = self.conv6_u(u5)
        u6 = self.upsample(u6)
        
        zero_6_1=torch.zeros([batch,32,320,320],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_6_2=torch.zeros([batch,48,320,320],dtype=x1.dtype, device=x1.device, requires_grad=True)
        
        
        
        conv1_2= torch.cat([conv1_2,zero_6_1], dim=1)
        conv0_2= torch.cat([conv0_2,zero_6_2], dim=1)
        u6 = u6+conv1_2+conv2_1+conv0_2 
        
        # # ### Block 7 ###
        u7 = self.conv7_u(u6)
        u7 = self.upsample(u7)
        
        zero_7_1=torch.zeros([batch,16,640,640],dtype=x1.dtype, device=x1.device, requires_grad=True)        
        
        conv0_1= torch.cat([conv0_1,zero_7_1], dim=1)
        
        u7 = u7+conv1_1+conv0_1
        
        out1=self.conv_last(u7)
        out2=self.conv_last1(u6)
        out3=self.conv_last2(u5)
        
        ## Scars Decodder  ####
        ## Scars Decodder  ####
        ## Scars Decodder  ####
        ## Scars Decodder  ####
        ## Scars Decodder  ####
        
        

        ### Block 1 ###
        u1_ = self.conv1_u(conv6_3)
        u1_ = self.upsample(u1_)
    
        u1_ = torch.cat([u1_,zero_1], dim=1)

        u1_ = u1_ + conv5_3 + conv6_2
        
        
        # # ### Block 2 ###
        u2_ = self.conv2_u(u1_)
        u2_ = self.upsample(u2_)
        

        
        u2_ = torch.cat([u2_,zero_2_2], dim=1)
       
        
        
        u2_ = u2_+ conv6_1+conv4_3+conv5_2
        
        # # ### Block 3 ###
        u3_ = self.conv3_u(u2_)
        u3_ = self.upsample(u3_)
        
     
        u3_ = u3_ + conv5_1+conv3_3+conv4_2

        
        # # ### Block 4 ###
        u4_ = self.conv4_u(u3_)
        u4_ = self.upsample(u4_)
        
     
        
        u4_ =  u4_+ conv4_1+conv2_3+conv3_2
        
        # # ### Block 5 ###
        u5_ = self.conv5_u(u4_)
        u5_ = self.upsample(u5_)
        
      
        
        
        u5_ = u5_+conv1_3+conv2_2+conv3_1+conv0_3
        
        # # ### Block 6 ###
        u6_ = self.conv6_u(u5_)
        u6_= self.upsample(u6_)
        

        u6_ = u6_+conv1_2+conv2_1+conv0_2 
        
        # # ### Block 7 ###
        u7_ = self.conv7_u(u6_)
        u7_ = self.upsample(u7_)
        

        
        u7_ = u7_+conv1_1+conv0_1
        
        out1_=self.conv_last_SC(u7_)
        out2_=self.conv_last1_SC(u6_)
        out3_=self.conv_last2_SC(u5_)
        
        return self.activation(out1),self.activation(out2),self.activation(out3),self.activation(out1_),self.activation(out2_),self.activation(out3_)
    
def model() -> m_unet:
    model = m_unet()
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(1, 640,640), (1, 320,320),(1,160,160)])
