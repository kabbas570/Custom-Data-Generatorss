
    

   
import numpy as np
from PIL import Image
import glob
import os
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
NUM_WORKERS=0
PIN_MEMORY=True
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



import cv2          
class CatDogDataset(Dataset):
    
    def __init__(self, imgs, transforms = None):
        
        super().__init__()
        self.imgs = imgs
        self.transforms = transforms
    
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        
        image_name = self.imgs[idx]
        #print(image_name)
        img = cv2.imread(image_name)
        img = cv2.resize(img,(224, 224),interpolation = cv2.INTER_AREA)
        
        if 'two' in image_name:
            label=torch.tensor([1,0,0])
        if 'three' in image_name:
             label=torch.tensor([0,1,0]) 
        if 'four' in image_name:
                  label=torch.tensor([0,0,1]) 
            
        return img,label
            
    
 
DIR_TRAIN= '/Users/kabbas570gmail.com/Documents/Breast cancer/data' ## apth to data folder
all_imgs=[]
for root,dirs,files in os.walk(DIR_TRAIN):

     for name in files:
         all_imgs.append(os.path.join(root,name))

         
train_dataset = CatDogDataset(all_imgs)

train_data_loader = DataLoader(
    dataset = train_dataset,
    num_workers = 0,
    batch_size = 2,pin_memory=True,
    shuffle = True
)

### give train_data_loader to model

### this part is for visulations ####
a=iter(train_data_loader)
a1=next(a)

batch_size=2

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

for i in range(batch_size):
    img=a1[0].numpy()[i,:,:,:]
    label=a1[1][i].numpy()
    
    #img=cv2.putText(img,str(label),org =(50, 50),font = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1

    # Using cv2.putText() method
    image = cv2.putText(img, str(label), org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    
    plt.figure()
    plt.imshow(image) 
    

   



      
         
         
         
         
         
         
         
         
