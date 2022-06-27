import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from scipy.ndimage import distance_transform_edt as distance

NUM_WORKERS=0
PIN_MEMORY=True


from skimage import segmentation as skimage_seg


def compute_sdf(img_gt, out_shape):
    T = 50
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = T*np.ones(out_shape) #np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                #sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return np.clip(normalized_sdf, -T, T)


class Dataset_(Dataset):
    def __init__(self, image_dir,mask_dir,b_dir,sc_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.b_dir=b_dir
        self.sc_dir=sc_dir
        self.images = os.listdir(image_dir)

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
        mask[np.where(mask>0)]=1.0
        
        
        
        boundry = np.load(bb_path,allow_pickle=True, fix_imports=True)
        boundry[np.where(boundry>0)]=1.0
        
        
        scars = np.load(sc_path,allow_pickle=True, fix_imports=True)
        scars[np.where(scars>0)]=1.0
        
        numpylabel_crop=mask
        numpylabel_crop_new = np.expand_dims(numpylabel_crop, 0)
        numpylabel_crop_new = np.expand_dims(numpylabel_crop_new, 0)
        numpylabel_crop_new = (numpylabel_crop_new>0)*1
        gt_dis = compute_sdf(numpylabel_crop_new, numpylabel_crop_new.shape)
        gt_LA_dis = np.squeeze(gt_dis, axis=1)
        
        
        
        if image.shape[0]==576:
         temp=np.zeros([640,640])
         temp1=np.zeros([640,640])
         temp2=np.zeros([640,640])
         temp3=np.zeros([640,640])
         
         temp[32:608, 32:608] = image
         image=temp
         
         temp1[32:608, 32:608] = mask
         mask=temp1
         
         temp2[32:608, 32:608] = scars
         scars=temp2
         
         temp3[32:608, 32:608] = boundry
         boundry=temp2
        

            
        image=np.expand_dims(image, axis=0)
        mask=np.expand_dims(mask, axis=0)
        boundry=np.expand_dims(boundry, axis=0)
        scars=np.expand_dims(scars, axis=0)
        gt_LA_dis=np.expand_dims(gt_LA_dis, axis=0)

        return image,mask,gt_LA_dis, boundry,scars,self.images[index][:-4]
    
def Data_Loader( test_dir,test_maskdir,b_dir,sc_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir,b_dir=b_dir,sc_dir=sc_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader
