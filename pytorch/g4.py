import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A

IMAGE_HEIGHT=384
IMAGE_WIDTH=384
NUM_WORKERS=0
PIN_MEMORY=True


transform1 = A.Compose([
    A.Resize(width=512, height=512)
])
transform2 = A.Compose([
    A.Resize(width=256, height=256)
])

transform3 = A.Compose([
    A.Resize(width=128, height=128)
])

class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir,transform1=transform1,transform2=transform2,transform3=transform3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

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
        
        
        if self.transform1 is not None:
            augmentations1 = self.transform1(image=image)
            augmentations2 = self.transform2(image=image,masks=[mask])
            augmentations3 = self.transform3(image=image)
            image1 = augmentations1["image"]
            image2 = augmentations2["image"]
            image3 = augmentations3["image"]
            mask = augmentations2["masks"][0]
            
            image1=np.expand_dims(image1, axis=0)
            image2=np.expand_dims(image2, axis=0)
            image3=np.expand_dims(image3, axis=0)
            mask=np.expand_dims(mask, axis=0)
           

        return image1,image2,image3,mask,self.images[index][:-4]
    
def Data_Loader( test_dir,test_maskdir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader








