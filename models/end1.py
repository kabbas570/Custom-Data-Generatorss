import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_patches(single_img,gt_sc):  ##wall---->[640,640],  img--->[640,640]  
    img_batch=[]
    gt_batch=[]
    for x in range(gt_sc.shape[0]):
        for y in range(gt_sc.shape[1]):
            if gt_sc[y,x]==1:
                img_crop=single_img[y-32:y+32,x-32:x+32]
                
                gt_sc_cropped=gt_sc[y-32:y+32,x-32:x+32]
                gt_sc_cropped=np.expand_dims(gt_sc_cropped,0)
                
                #img_crop=img_crop.cpu()
                #img_crop=img_crop.numpy()
                            
                mean=np.mean(img_crop,keepdims=True)
                std=np.std(img_crop,keepdims=True)
                img_crop=(img_crop-mean)/std
                            
                gen_img=np.zeros([3,img_crop.shape[0],img_crop.shape[1]]) 
                    
                img_o=img_crop.copy()    ## orignal img
                    
                img_crop[np.where(img_crop<0)]=0   #### no negatives
                    
                hitss=np.histogram(img_crop, bins=5)
                    
                value_=hitss[1][1]
                img1=np.zeros([img_crop.shape[0],img_crop.shape[1]])  
                img1[np.where(img_crop>=value_)]=1  
                new_img=img1*img_o                 #### from hists
                    
                gen_img[0,:,:]=img_o
                gen_img[1,:,:]=img_crop
                gen_img[2,:,:]=new_img
                        
                #gen_img=torch.tensor(gen_img,dtype=torch.float)
                #gen_img=torch.unsqueeze(gen_img, 0)
                
                img_batch.append(gen_img)
                gt_batch.append(gt_sc_cropped)
    
    img_batch=np.array(img_batch)
    gt_batch=np.array(gt_batch)
    
    return img_batch,gt_batch

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
def double_conv01(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )

def double_conv11(in_channels, out_channels,f_size,p_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 

def double_conv_u1(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 


def trans_1(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
       nn.ConvTranspose2d(in_channels,out_channels, kernel_size=f_size, stride=st_size),
       #nn.BatchNorm2d(num_features=out_channels),
       #nn.ReLU(inplace=True),
    ) 



class m_unet4(nn.Module):
    def __init__(self):
        super(m_unet4, self).__init__()

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        
        
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)
        self.activation = torch.nn.Sigmoid()
        
        self.up_ = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        
        self.dconv_down1 = double_conv01(3, 64,(3,3))
        self.dconv_down2 = double_conv11(64, 128,(3,3),(1,1))
        self.dconv_down3 = double_conv11(128, 256,(3,3),(1,1))
        self.dconv_down4 = double_conv01(256, 512,(3,3))
        self.dconv_down5 = double_conv01(512, 512,(3,3))
        
        #self.up0 = trans_1(512,256, 2,2)
        self.up11 = trans_1(256,256,  2,2)
        self.up22 = trans_1(128, 128, 2,2)
        #self.up3 = trans_1(128, 64, 2,2)
        
        
        self.m = nn.Dropout(p=0.10)

        
        self.dconv_up0 = double_conv_u1(512, 512,(3,3))
        self.dconv_up1 = double_conv_u1(512 + 512, 512,(3,3))
        self.dconv_up2 = double_conv_u1(512+256, 256,(3,3))
        self.dconv_up3 = double_conv_u1(256+128,128,(3,3))
        
        self.dconv_up4 = double_conv_u1(192,64,(3,3))
        
        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, img,sc_gt):
        ##encoder 1 ##
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        ##decoder###
         
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_LA = self.outc(x)
        
        img=img.cpu().numpy()
        sc_gt=sc_gt.cpu().numpy()
        
        out2=[]
        for q in range(img.shape[0]):
          #single_la=out1[k,0,:,:]    ##-->[640,640]
          single_img=img[q,0,:,:]   ##-->[640,640]
          single_sc_gt=sc_gt[q,0,:,:]  ##-->[640,640]
          
          print(single_img.shape)
          print(single_sc_gt.shape)
          
          if np.sum(single_sc_gt)!=0:
            img_batch,gt_batch=make_patches(single_img,single_sc_gt)
            print(img_batch.shape)
            
          else:
              img_batch=torch.zeros([5,3,64,64])
              gt_batch=torch.zeros([5,1,64,64])
              print('not printed')
               
          img_batch=torch.tensor(img_batch)
          gt_batch=torch.tensor(gt_batch)
          img_batch = img_batch.to(device=DEVICE,dtype=torch.float)
          gt_batch = gt_batch.to(device=DEVICE,dtype=torch.float)
          
          
            
          conv1 = self.dconv_down1(img_batch)      
          conv2 = self.dconv_down2(conv1)
          conv3 = self.dconv_down3(conv2)
          conv4 = self.dconv_down4(conv3)
          conv5 = self.dconv_down5(conv4)

            # # ## decoder ####
            
          conv5=self.m(conv5)
          u0=self.dconv_up0(conv5)
          u0 = torch.cat([u0, conv4], dim=1) 

          u1=self.dconv_up1(u0)
            
          u1 = torch.cat([u1, conv3], dim=1) 
            
          u2=self.dconv_up2(u1)
          u2=self.up11(u2)
          u2 = torch.cat([u2, conv2], dim=1) 
            
            
          u3=self.dconv_up3(u2)
          u3=self.up22(u3)
          u3 = torch.cat([u3, conv1], dim=1) 
          u3=self.dconv_up4(u3)
            
          out=self.conv_last(u3)
          out2.append(out)
          
        out2 = torch.cat(out2, dim=0)
        return self.activation(x_LA) ,self.activation(out2) 
        
        
        
model1 = m_unet4().to(device=DEVICE,dtype=torch.float)



img1=np.load(r'C:\My data\sateg0\task_1_both_data\task1_2d\train\img\train_1_2.npy')
gt1=np.load(r'C:\My data\sateg0\task_1_both_data\task1_2d\train\sc_gt\train_1_2sc_gt.npy')


img=np.zeros([2,576,576])
gt=np.zeros([2,576,576])

img[0,:,:]=img1
img[1,:,:]=img1

gt[0,:,:]=gt1
gt[1,:,:]=gt1

img=torch.tensor(img)
gt=torch.tensor(gt)
img=torch.unsqueeze(img, 1)

print(img.shape)
#img=torch.unsqueeze(img, 0)

gt=torch.unsqueeze(gt, 1)
#gt=torch.unsqueeze(gt, 0)


img = img.to(device=DEVICE,dtype=torch.float)
gt = gt.to(device=DEVICE,dtype=torch.float)
out1,out2 = model1(img,gt)

print(out2.shape)



