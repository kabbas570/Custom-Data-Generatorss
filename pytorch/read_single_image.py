import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from PIL import Image


IMAGE_HEIGHT=384
IMAGE_WIDTH=384
NUM_WORKERS=0
PIN_MEMORY=True

transform = A.Compose([
    A.RandomCrop(width=IMAGE_HEIGHT, height=IMAGE_WIDTH)
])


import cv2
class Dataset_(Dataset):
    def __init__(self, image_dir,transform=transform):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
       
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ## reading image ###
        img_path = os.path.join(self.image_dir, self.images[index])
        #image = Image.open(img_path)

        image = np.array(Image.open(img_path).convert('RGB'))
        ### read clinincal data ###
        ### normalize the age ###
        
        if self.transform is not None:
            image = self.transform(image=image)
            
        return image,self.images[index]
        #return image,c1,c2,gt, self.transform(self.images[index])

    
def Data_Loader( test_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


images_folder = r'C:\Users\Abbas Khan\Downloads\train\image'

loader=Data_Loader(images_folder,1)
a=iter(loader)
a1=next(a)
