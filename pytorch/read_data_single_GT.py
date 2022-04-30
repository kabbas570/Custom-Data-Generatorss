import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


IMAGE_HEIGHT=224
IMAGE_WIDTH=224
BATCH_SIZE=5
TRAIN_IMG_DIR='/data/home/acw676/Challenge/data/train/img/'
TRAIN_MASK_DIR='/data/home/acw676/Challenge/data/train/gt/'

VAL_IMG_DIR='/data/home/acw676/Challenge/data/valid/img/'
VAL_MASK_DIR='/data/home/acw676/Challenge/data/valid/gt/'

TEST_IMG_DIR='/Users/kabbas570gmail.com/Documents/Challenge/valid/img'
TEST_MASK_DIR='/Users/kabbas570gmail.com/Documents/Challenge/valid/gt'
TEST_BATCH_SIZE=1


NUM_WORKERS=0
PIN_MEMORY=True

train_transforms = A.Compose(
        [
            #ToTensorV2(),
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35),
            #A.HorizontalFlip,
            #A.HorizontalFlip(),
            #A.VerticalFlip,
            #A.Rotate(limit=35, p=1.0),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
        [   
            #ToTensorV2(),
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.HorizontalFlip(),
            #A.VerticalFlip,
            #A.Rotate(limit=35, p=1.0),
            ToTensorV2(),
        ],
    )
test_transforms = A.Compose(
        [   
            #ToTensorV2(),
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.HorizontalFlip(),
            #A.VerticalFlip,
            #A.Rotate(limit=35, p=1.0),
            ToTensorV2(),
        ],
    )



def normalize_(img): # img=[576,576,1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tr = transform(img)
    mean, std = img_tr.mean([1,2]), img_tr.std([1,2]) # shape of img_tr=[1,256,256]
    transform_norm = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
      
    img_nor = transform_norm(img_tr) ## get this 
    return img_nor

class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'_gt.npy')
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        
        max_=image.max()
        image=image/max_

        #image = image.unsqueeze(0)
        #image=np.expand_dims(image, axis=2)
        #image=normalize_(image)
        #image=np.array(image)
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        #mask=np.expand_dims(mask, axis=0)
        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask=np.expand_dims(mask, axis=0)

        return image, mask
    
def get_loaders(
    train_dir=TRAIN_IMG_DIR,
    train_maskdir=TRAIN_MASK_DIR,
    val_dir=VAL_IMG_DIR,
    val_maskdir=VAL_MASK_DIR,
    batch_size=BATCH_SIZE,
    train_transform=train_transforms,
    val_transform=val_transforms,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    train_ds = Dataset_(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Dataset_(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader



def test_data(
    test_dir=TEST_IMG_DIR,
    test_maskdir=TEST_MASK_DIR,
    
    batch_size=1,
    test_transform=test_transforms,

    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    test_ids = Dataset_(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return test_loader

















