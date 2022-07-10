import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import cv2

NUM_WORKERS=0
PIN_MEMORY=True

transform2 = A.Compose([
    A.Resize(width=320, height=320)
])

class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir,transform2=transform2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform2 = transform2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'_gt.npy')
      
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        mask[np.where(mask>0)]=1.0
        #print(image.shape[0])
        if image.shape[0]==576:
          temp=np.zeros([640,640])
          temp1=np.zeros([640,640])
          
          temp[32:608, 32:608] = image
          image=temp
          
          temp1[32:608, 32:608] = mask
          mask=temp1
          
        h, w = mask.shape
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=5)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        mask1= mask - mask_erode
        
        if self.transform2 is not None:
            
            augmentations2 = self.transform2(image=image)

            image2 = augmentations2["image"]

            image=np.expand_dims(image, axis=0)
            image2=np.expand_dims(image2, axis=0)
            mask=np.expand_dims(mask, axis=0)
            mask1=np.expand_dims(mask1, axis=0)
          

        return image,image2,mask,mask1,self.images[index][:-4]
    
def Data_Loader( test_dir,test_maskdir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader
