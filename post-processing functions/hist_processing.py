import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import matplotlib.pyplot as plt

NUM_WORKERS=0
PIN_MEMORY=True

transform = A.Compose([
    A.Resize(width=128, height=192)
])

def normalize(x):

    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))



class Dataset_v(Dataset):
    def __init__(self, image_dir, mask_dir,transform=transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir,self.images[index][:-4]+'_gt.npy')
       
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        mask[np.where(mask>0)]=1.0
        
        # image=normalize(new_img)
        
        # mean=np.mean(image)
        # std=np.std(image)
        # image=(image-mean)/std
        

        
        # scar_only=image*mask
        # max_=np.max(scar_only)
        # scar_only[np.where(scar_only==0)]=max_
        # min_=np.min(scar_only)
        # max_=np.max(scar_only)
        
        # img1=np.zeros([image.shape[0],image.shape[1]])
        # img1[np.where((image>=min_) & (image<=max_))]=1
        # new_img=img1*image
        
        hitss=np.histogram(image, bins=5)
        stacked_img=np.zeros([image.shape[0],image.shape[1],3])
        for k in range(2,5):
            value_=hitss[1][k]
            img1=np.zeros([image.shape[0],image.shape[1]])  
            img1[np.where(image>=value_)]=1  
            new_img=img1*image
            
            
            new_img=normalize(new_img)
            stacked_img[:,:,k-2] =new_img
        
        image=stacked_img
            
        

        #image=np.expand_dims(stacked_img, axis=0)
        
        
        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            
            image = augmentations["image"]
            mask = augmentations["mask"]
        

            mask=np.expand_dims(mask, axis=0)
            image=np.transpose(image, (2, 0, 1))
        
           
        return image,mask,self.images[index]
    
def Data_Loader_v( test_dir,test_maskdir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_v( image_dir=test_dir, mask_dir=test_maskdir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader

val_imgs=r'C:\Users\Abbas Khan\Desktop\PROS\nw_data\val1\img'
val_masks=r'C:\Users\Abbas Khan\Desktop\PROS\nw_data\val1\gt'
val_loader=Data_Loader_v(val_imgs,val_masks,batch_size=1)

a=iter(val_loader)

# a1=next(a)
# img=a1[0][0,0,:,:].numpy()
# gt=a1[1][0,0,:,:].numpy()
    
for i in range(4):
    a1=next(a)
    img=a1[0][0,:,:,:].numpy()
    gt=a1[1][0,0,:,:].numpy()
    
    #img[np.where(gt==1)]=0
    

    img1=np.transpose(img, (1,2,0))
    
    plt.figure()
    plt.imshow(img1)
        
