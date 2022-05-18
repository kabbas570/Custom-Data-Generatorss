import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import nibabel as nib


IMAGE_HEIGHT=384
IMAGE_WIDTH=384
NUM_WORKERS=0
PIN_MEMORY=True


transform = A.Compose([
    A.Resize(width=IMAGE_HEIGHT, height=IMAGE_WIDTH)
])

class Dataset_(Dataset):
    def __init__(self, image_dir, transform=transform):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]+'/enhanced.nii.gz')
        mask_path = os.path.join(self.image_dir, self.images[index]+'/atriumSegImgMO.nii.gz')
        scar_path = os.path.join(self.image_dir, self.images[index]+'/scarSegImgM.nii.gz')
        
        image =nib.load(img_path).get_fdata()
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        mask =nib.load(mask_path).get_fdata()
        mask[np.where(mask>0)]=1.0
        
        scar =nib.load(scar_path).get_fdata()
        scar[np.where(scar>0)]=1.0
        
        
        if self.transform is not None:
            augmentations = self.transform(image=image,masks=[mask,scar])
            image = augmentations["image"]
            mask = augmentations["masks"][0]
            scar = augmentations["masks"][1]
            #image=np.moveaxis(image,[0,1,2,3],[0,2,1,3])
            image=np.transpose(image, (2, 0, 1))
            mask=np.transpose(mask, (2, 0, 1))
            scar=np.transpose(scar, (2, 0, 1))

        return image,mask,scar,self.images[index]
    
def Data_Loader( test_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader

batch_size=1
image_path = '/Users/kabbas570gmail.com/Downloads/dataset/task1/train_data'

val_loader=Data_Loader(image_path,batch_size)

a=iter(val_loader)
a1=next(a)

img=a1[0].numpy()[0,27,:,:]
g1=a1[1].numpy()[0,27,:,:]
g2=a1[2].numpy()[0,27,:,:]
