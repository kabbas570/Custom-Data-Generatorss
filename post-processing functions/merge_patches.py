import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob




# from_main_model=np.load(r'C:\My data\sateg0\valid\sc_gt/train_7_30sc_gt.npy')
# from_main_model[np.where(from_main_model>0)]=1

from_main_model=np.load(r'C:\My data\sateg0\valid\b_gt/train_7_30b_gt.npy')
from_main_model[np.where(from_main_model>0)]=1

org_img=np.load(r'C:\My data\sateg0\valid\img/train_7_30.npy')



merged_pred=np.zeros([576,576])
for x in range(from_main_model.shape[0]):
   for y in range(from_main_model.shape[1]):
       if from_main_model[y,x]==1.0:
           crop_img=org_img[y-24:y+24,x-24:x+24]
           
           #model_output=model(img_crop)
           
           patch_pred=np.load(r'C:\My data\temp_data\patches\valid\gt/train_7_30_81.npy')
           
           merged_pred[y-24:y+24,x-24:x+24]=  patch_pred

           
           
           
           
           
