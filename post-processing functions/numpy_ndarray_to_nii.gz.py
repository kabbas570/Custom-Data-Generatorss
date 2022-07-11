from tqdm import tqdm
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import nibabel as nib


gt_f=r'C:\My data\sateg0\task_1_both_data\val_data\test_0\enhanced.nii.gz'
gt1  = nib.load(gt_f).get_fdata()

gt_1=gt1[:,:,20]




img = nib.Nifti1Image(gt1, np.eye(4))  # Save axis for data (just identity)


savedir=r'C:\My data\sateg0\task_1_both_data\val_data'
name='train_1'
gt_ = nib.Nifti1Image(gt1, img.affine, img.header)

dir_name2=savedir+'/'+ name   ## dir name
os.makedirs(dir_name2)
nib.save(gt_,dir_name2 + '/enhanced.nii.gz')



img.to_filename(os.path.join(savedir,'test4d.nii.gz'))  # Save as NiBabel file



gen=nib.load(r'C:\My data\sateg0\task_1_both_data\val_data\test4d.nii.gz').get_fdata()

gen1=gen[:,:,20]
