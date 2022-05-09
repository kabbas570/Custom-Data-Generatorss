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
    A.RandomCrop(width=IMAGE_HEIGHT, height=IMAGE_WIDTH)
])

class Dataset_(Dataset):
    def __init__(self, image_dir,transform=transform):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
       
       
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        
        
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image,self.images[index][:-4]
    
def Data_Loader( test_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader










