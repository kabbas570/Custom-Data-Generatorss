import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import nibabel as nib


NUM_WORKERS=0
PIN_MEMORY=True


transform2 = A.Compose([
    A.Resize(width=320, height=320)
])

class Dataset_(Dataset):
    def __init__(self, image_dir,transform2=transform2):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform2 = transform2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]+'/enhanced.nii.gz')
        mask_path = os.path.join(self.image_dir, self.images[index]+'/atriumSegImgMO.nii.gz')
        
        
        image =nib.load(img_path).get_fdata()
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        mask =nib.load(mask_path).get_fdata()
        mask[np.where(mask>0)]=1.0
        
        if image.shape[0]==576:
            temp=np.zeros([640,640,44])
            
                  
            temp[32:608, 32:608,:] = image
            image=temp
            
            
            temp1=np.zeros([640,640,44])    
            temp1[32:608, 32:608,:] = mask
            mask=temp1
            
        if self.transform2 is not None:
            
            augmentations2 = self.transform2(image=image)

            image2 = augmentations2["image"]
            
            print(image2.shape)
            
        
            #image2=np.moveaxis(image2, [0,1,2],[2,1,0])
            
            image2=np.transpose(image2, (2, 0, 1))
            image=np.transpose(image, (2, 0, 1))
            mask=np.transpose(mask, (2, 0, 1))
                
            #image = np.expand_dims(image, axis=0)
           # image2 = np.expand_dims(image2, axis=0)
            #mask = np.expand_dims(mask, axis=0)
            
           

        return image,image2,mask,self.images[index]
    
def Data_Loader( test_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader



batch_size=1

train_data=r'C:\My data\sateg0\task_2_both_data\task2_3D\test'

train_loader=Data_Loader(train_data,batch_size)


a=iter(train_loader)
a1=next(a)

import torch
img=a1[0]
print(img.shape)
img=torch.moveaxis(img, [0,1,2,3],[1,0,2,3])
print(img.shape)
