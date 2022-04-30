import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from skimage.transform import resize
IMAGE_HEIGHT=384
IMAGE_WIDTH=384
BATCH_SIZE=2
NUM_WORKERS=0
PIN_MEMORY=True

class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir,b_dir,sc_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.b_dir=b_dir
        self.sc_dir=sc_dir
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.bb = os.listdir(b_dir)
        self.sc = os.listdir(sc_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'_gt.npy')
        bb_path = os.path.join(self.b_dir, self.images[index][:-4]+'b_gt.npy')
        sc_path = os.path.join(self.sc_dir, self.images[index][:-4]+'sc_gt.npy')
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        boundry = np.load(bb_path,allow_pickle=True, fix_imports=True)
        scars = np.load(sc_path,allow_pickle=True, fix_imports=True)
        
        image=resize(image,(IMAGE_HEIGHT,IMAGE_WIDTH))
        mask=resize(mask,(IMAGE_HEIGHT,IMAGE_WIDTH))
        boundry=resize(boundry,(IMAGE_HEIGHT,IMAGE_WIDTH))
        scars=resize(scars,(IMAGE_HEIGHT,IMAGE_WIDTH))
        
        mask[np.where(mask!=0)]=1
        boundry[np.where(boundry!=0)]=1
        scars[np.where(scars!=0)]=1
        
        image=np.expand_dims(image, axis=0)
        mask=np.expand_dims(mask, axis=0)
        boundry=np.expand_dims(boundry, axis=0)
        scars=np.expand_dims(scars, axis=0)
        return image,mask,boundry,scars
    
def Data_Loader(
    test_dir,
    test_maskdir,
    b_dir,
    sc_dir,
    batch_size,

    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    test_ids = Dataset_(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        b_dir=b_dir,
        sc_dir=sc_dir,
    )

    test_loader = DataLoader(
        test_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return test_loader










