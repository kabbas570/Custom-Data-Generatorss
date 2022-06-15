
import torch
from torch import nn

class Spatial_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=7,padding=7//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x1,x2,x3) :
        max_result1,_=torch.max(x1,dim=1,keepdim=True)
        avg_result1=torch.mean(x1,dim=1,keepdim=True)
        
        
        
        max_result2,_=torch.max(x2,dim=1,keepdim=True)
        avg_result2=torch.mean(x2,dim=1,keepdim=True)
        
        
        max_result3,_=torch.max(x3,dim=1,keepdim=True)
        avg_result3=torch.mean(x3,dim=1,keepdim=True)
        
        Resultant_max= max_result1+max_result2+max_result3
        Resultant_avg=avg_result1+avg_result2+avg_result3
        
        result=torch.cat([Resultant_avg,Resultant_max],1)

        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class Attention_Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.A1=Spatial_1()


    def forward(self, x1,x2,x3):
        residual1=x1
        residual2=x2
        residual3=x3
        
        out=self.A1(x1,x2,x3)
        
        out1=out*residual1
        out2=out*residual2
        out3=out*residual3

        return out1,out2,out3
    
class Spatial_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=7,padding=7//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x1,x2) :
        max_result1,_=torch.max(x1,dim=1,keepdim=True)
        avg_result1=torch.mean(x1,dim=1,keepdim=True)
        
        
        
        max_result2,_=torch.max(x2,dim=1,keepdim=True)
        avg_result2=torch.mean(x2,dim=1,keepdim=True)
        
        
     
        
        Resultant_max= max_result1+max_result2
        Resultant_avg=avg_result1+avg_result2
        
        result=torch.cat([Resultant_avg,Resultant_max],1)

        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class Attention_Block_2(nn.Module):

    def __init__(self):
        super().__init__()
        self.A1=Spatial_2()


    def forward(self, x1,x2):
        residual1=x1
        residual2=x2
        
        out=self.A1(x1,x2)
        
        out1=out*residual1
        out2=out*residual2

        return out1,out2

class Spatial_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=7,padding=7//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x1,x2,x3,x4):
        
        max_result1,_=torch.max(x1,dim=1,keepdim=True)
        avg_result1=torch.mean(x1,dim=1,keepdim=True)
        
        
        
        max_result2,_=torch.max(x2,dim=1,keepdim=True)
        avg_result2=torch.mean(x2,dim=1,keepdim=True)
        
        
        max_result3,_=torch.max(x3,dim=1,keepdim=True)
        avg_result3=torch.mean(x3,dim=1,keepdim=True)
        
        max_result4,_=torch.max(x4,dim=1,keepdim=True)
        avg_result4=torch.mean(x4,dim=1,keepdim=True)
        
        
        Resultant_max= max_result1+max_result2+max_result3+max_result4
        Resultant_avg=avg_result1+avg_result2+avg_result3+avg_result4
        
        result=torch.cat([Resultant_avg,Resultant_max],1)
        
        
     
        
        Resultant_max= max_result1+max_result2
        Resultant_avg=avg_result1+avg_result2
        
        result=torch.cat([Resultant_avg,Resultant_max],1)

        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class Attention_Block_4(nn.Module):

    def __init__(self):
        super().__init__()
        self.A1=Spatial_4()


    def forward(self, x1,x2,x3,x4):
        residual1=x1
        residual2=x2
        residual3=x3
        residual4=x4
        out=self.A1(x1,x2,x3,x4)
        
        out1=out*residual1
        out2=out*residual2
        
        out3=out*residual3
        out4=out*residual4
        

        return out1,out2,out3,out4
    
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
        
        self.conv0=conv1_(1,16)
        self.conv1=conv1_(16,32)
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
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.activation = torch.nn.Sigmoid()
        
        ## res paths ###
        
        self.att_1=Attention_Block_2()
        self.att_2=Attention_Block()
        self.att_3=Attention_Block()
        self.att_4=Attention_Block()
        self.att_5=Attention_Block_4()
        self.att_6=Attention_Block()
        self.att_7=Attention_Block_2()

    def forward(self,x1,x2,x3):
        
        ### Encoder Design ###
           
           
           ## conv0  ###
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
        
        
        ## conv6  ###
        conv6_1 = self.conv6(conv5_1)
        conv6_2 = self.conv6(conv5_2)
        conv6_3 = self.conv6(conv5_3) 
        conv6_1 = self.maxpool(conv6_1)
        conv6_2 = self.maxpool(conv6_2)
        conv6_3 = self.maxpool(conv6_3)
        
        
        
       #        ### Decoder Design ###
#
        batch=conv6_1.shape[0]

        ### Block 1 ###
        u1 = self.conv1_u(conv6_3)
        u1 = self.upsample(u1)
        
        
        conv5_3,conv6_2=self.att_1(conv5_3,conv6_2)
    
        zero_1=torch.zeros([batch,512,10,10],dtype=x1.dtype, device=x1.device, requires_grad=True)
        u1 = torch.cat([u1,zero_1], dim=1)
        conv5_3=torch.cat([conv5_3,zero_1], dim=1)

        u1 = u1 + conv5_3 + conv6_2
        
        
        # # ### Block 2 ###
        u2 = self.conv2_u(u1)
        u2 = self.upsample(u2)
        
        
        conv6_1,conv4_3,conv5_2=self.att_2(conv6_1,conv4_3,conv5_2)
        
        zero_2_1=torch.zeros([batch,768,20,20],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_2_2=torch.zeros([batch,512,20,20],dtype=x1.dtype, device=x1.device, requires_grad=True)
        

        
        u2 = torch.cat([u2,zero_2_2], dim=1)
        conv4_3 = torch.cat([conv4_3,zero_2_1], dim=1)
        conv5_2 = torch.cat([conv5_2,zero_2_2], dim=1)
        
        
        u2 = u2+ conv6_1+conv4_3+conv5_2
        
        # # ### Block 3 ###
        u3 = self.conv3_u(u2)
        u3 = self.upsample(u3)
        
        conv5_1,conv3_3,conv4_2=self.att_3(conv5_1,conv3_3,conv4_2)

        
        zero_3_1=torch.zeros([batch,384,40,40],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_3_2=torch.zeros([batch,256,40,40],dtype=x1.dtype, device=x1.device, requires_grad=True)
        
        
        #u3 = torch.cat([u3,zero_3_2], dim=1)
        conv3_3 = torch.cat([conv3_3,zero_3_1], dim=1)
        conv4_2 = torch.cat([conv4_2,zero_3_2], dim=1)
        

         
        u3 = u3+ conv5_1+conv3_3+conv4_2

        
        # # ### Block 4 ###
        u4 = self.conv4_u(u3)
        u4 = self.upsample(u4)
        
        
        conv4_1,conv2_3,conv3_2=self.att_4(conv4_1,conv2_3,conv3_2)

        
        zero_4_1=torch.zeros([batch,192,80,80],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_4_2=torch.zeros([batch,128,80,80],dtype=x1.dtype, device=x1.device, requires_grad=True)
        
        
        
        #u4= torch.cat([u4,zero_4_1], dim=1)
        conv2_3 = torch.cat([conv2_3,zero_4_1], dim=1)
        conv3_2 = torch.cat([conv3_2,zero_4_2], dim=1)

        
        u4 =  u4+ conv4_1+conv2_3+conv3_2
        
        # # ### Block 5 ###
        u5 = self.conv5_u(u4)
        u5 = self.upsample(u5)
        
        conv1_3,conv2_2,conv3_1,conv0_3=self.att_5(conv1_3,conv2_2,conv3_1,conv0_3)

        
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
        
        conv1_2,conv2_1,conv0_2=self.att_6(conv1_2,conv2_1,conv0_2)


        zero_6_1=torch.zeros([batch,32,320,320],dtype=x1.dtype, device=x1.device, requires_grad=True)
        zero_6_2=torch.zeros([batch,48,320,320],dtype=x1.dtype, device=x1.device, requires_grad=True)
        
        
        
        conv1_2= torch.cat([conv1_2,zero_6_1], dim=1)
        conv0_2= torch.cat([conv0_2,zero_6_2], dim=1)
        u6 = u6+conv1_2+conv2_1+conv0_2 
        
        # # ### Block 7 ###
        u7 = self.conv7_u(u6)
        u7 = self.upsample(u7)
        
        conv1_1,conv0_1=self.att_7(conv1_1,conv0_1)


        zero_7_1=torch.zeros([batch,16,640,640],dtype=x1.dtype, device=x1.device, requires_grad=True)        
        
        conv0_1= torch.cat([conv0_1,zero_7_1], dim=1)
        
        u7 = u7+conv1_1+conv0_1
        
        out1=self.conv_last(u7)
        out2=self.conv_last1(u6)
        out3=self.conv_last2(u5)
        
        return self.activation(out1),self.activation(out2),self.activation(out3)
    
def model() -> m_unet:
    model = m_unet()
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from torchsummary import summary
model = model()
model.to(device=DEVICE,dtype=torch.float)
summary(model, [(1, 640,640), (1, 320,320),(1,160,160)])

