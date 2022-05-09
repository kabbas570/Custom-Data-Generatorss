from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision
import numpy as np
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




## single img and multiple GT ### 
batch_size=1
image_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/img'
mask_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/seg_gt/'
b_path='/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/b_gt/'
sc_path='/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/sc_gt/'

from g1 import Data_Loader
val_loader=Data_Loader(image_path,mask_path,b_path,sc_path,batch_size)

a=iter(val_loader)
a1=next(a)

img=a1[0].numpy()
g1=a1[1].numpy()
g2=a1[2].numpy()
g3=a1[3].numpy()

img=img[0,0,:,:]
g1=g1[0,0,:,:]
g2=g2[0,0,:,:]
g3=g3[0,0,:,:]


## single img and single GT ###

batch_size=1
image_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/img'
mask_path = '/Users/kabbas570gmail.com/Documents/Challenge/testing/data/valid1/seg_gt/'

from g2 import Data_Loader
val_loader=Data_Loader(image_path,mask_path,batch_size)

a=iter(val_loader)
a1=next(a)

img=a1[0].numpy()
g1=a1[1].numpy()


img=img[0,0,:,:]
g1=g1[0,0,:,:]