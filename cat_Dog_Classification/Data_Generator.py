import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import cv2
import torch

IMAGE_HEIGHT=384
IMAGE_WIDTH=384
NUM_WORKERS=0
PIN_MEMORY=True


# transform = A.Compose([
#     A.Resize(width=IMAGE_HEIGHT, height=IMAGE_WIDTH)
# ])

class Dataset_(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        
        image = cv2.imread(img_path)
        image = cv2.resize(image,(224, 224),interpolation = cv2.INTER_AREA)
        
        #image = image/255

        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        image = np.moveaxis(image,2, 0)
        
        image_name = self.images[index]
        
        if 'cat' in image_name:
            label=torch.tensor([1])
        if 'dog' in image_name:
             label=torch.tensor([0]) 
            
        return image,label
    
def Data_Loader( img_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_( image_dir=img_dir)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

# batch_size = 4
# data_path = r'C:\My_Data\mahmud\train'

# train_loader=Data_Loader(data_path,batch_size)

# a=iter(train_loader)
# a1=next(a)

# # img=a1[0].numpy()
# # gt1=a1[1].numpy()[0,0,:,:]
# # gt2=a1[1].numpy()[0,1,:,:]