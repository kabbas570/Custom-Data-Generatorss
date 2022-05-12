import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A

IMAGE_HEIGHT=384
IMAGE_WIDTH=384
NUM_WORKERS=0
PIN_MEMORY=True


transform = A.Compose([
    A.Resize(width=IMAGE_HEIGHT, height=IMAGE_WIDTH)
])

class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir,sc_dir,transform=transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sc_dir=sc_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'_gt.npy')
        sc_path = os.path.join(self.sc_dir, self.images[index][:-4]+'sc_gt.npy')
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        
        
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        mask[np.where(mask>0)]=1.0
        
        
        
        
        scars = np.load(sc_path,allow_pickle=True, fix_imports=True)
        scars[np.where(scars>0)]=1.0
        

        
        
        if self.transform is not None:
            augmentations = self.transform(image=image,masks=[mask,scars])
            image = augmentations["image"]
            mask = augmentations["masks"][0]
            scars = augmentations["masks"][1]
            
            gt=np.zeros([2,IMAGE_HEIGHT,IMAGE_WIDTH])
            gt[0,:,:][np.where(mask>0)]=1.0
            gt[1,:,:][np.where(scars>0)]=1.0
            
            image=np.expand_dims(image, axis=0)
        
        
        return image,gt,self.images[index][:-4]
    
def Data_Loader( test_dir,test_maskdir,sc_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir,sc_dir=sc_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader




batch_size=1
image_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/img'
mask_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/seg_gt/'
sc_path='/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/sc_gt/'

val_loader=Data_Loader(image_path,mask_path,sc_path,batch_size)

a=iter(val_loader)
a1=next(a)

img=a1[0].numpy()
gt1=a1[1].numpy()[0,0,:,:]
gt2=a1[1].numpy()[0,1,:,:]






