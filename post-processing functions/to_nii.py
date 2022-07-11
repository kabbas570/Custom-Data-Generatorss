import numpy as np
import nibabel as nib


savedir=r'C:\My data\cada\task2'
fileNPY1 = r"C:\My data\cada\task2\test\gt\train_1_34_gt.npy"
img_array1 = np.load(fileNPY1) 
a1 = nib.Nifti1Image(img_array1 ,  nibimage.affine, nibimage.header)

nib.save(a1, savedir + '/a1.nii.gz')


c1  = nib.load(r'C:\My data\cada\task2\task2_processed to 44\train_10/enhanced.nii.gz').get_fdata()

n1=c1[:,:,20]

o1  = nib.load(r'C:\Users\Abbas Khan\Downloads\dataset\task2\train_data\train_5/atriumSegImgMO.nii.gz').get_fdata()

a1=o1[:,:,25]



### convert 88 to 44  #####


import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 


files_names=[]

img_f=[]
gt_f=[]
sc_f=[]
for root, dirs, files in os.walk(r"C:\Users\Abbas Khan\Downloads\dataset\task2"):
   for name in files:
      files_names.append(os.path.join(root, name))
      if name=='enhanced.nii.gz':
          img_f.append(os.path.join(root, name)) 
      if name=='scarSegImgM.nii.gz':
          sc_f.append(os.path.join(root, name))
      if name=='atriumSegImgMO.nii.gz':
          gt_f.append(os.path.join(root, name)) 


savedir=r'C:\My data\cada\task2\task2_processed to 44'
for i in range(130):
    
    nii_img  = nib.load(img_f[i]).get_fdata()

    if nii_img.shape[2]==44:    ## for 44 go and save again
        nii_img  = nib.load(img_f[i])
        nii_gt  = nib.load(gt_f[i])
        name_img=img_f[i][55:-16]
        dir_name=savedir+'/'+name_img   ## dir name
        os.makedirs(dir_name)
        nib.save(nii_img, dir_name + '/enhanced.nii.gz')
        nib.save(nii_gt,dir_name + '/atriumSegImgMO.nii.gz')
    
    if nii_img.shape[2]==88:
        
        nibimage  = nib.load(img_f[i])
        nibgt  = nib.load(gt_f[i])
        
        nii_img=nibimage.get_fdata()
        nii_gt=nibgt.get_fdata()
        
        name_img=img_f[i][55:-16]
        
        dir_name1=savedir+'/'+name_img   ## dir name
        os.makedirs(dir_name1)
        
        
        name_img=name_img+'_T'
        #print(name_img)
        
        dir_name2=savedir+'/'+name_img   ## dir name
        os.makedirs(dir_name2)
        
        
        first_img=nii_img[:,:,0:44]
        second_img=nii_img[:,:,44:88]
        
        first_gt=nii_gt[:,:,0:44]
        second_gt=nii_gt[:,:,44:88]
        
        first_img = nib.Nifti1Image(first_img ,  nibimage.affine, nibimage.header)
        second_img = nib.Nifti1Image(second_img ,  nibimage.affine, nibimage.header)
        
        first_gt = nib.Nifti1Image(first_gt ,  nibgt.affine, nibgt.header)
        second_gt = nib.Nifti1Image(second_gt ,  nibgt.affine, nibgt.header)
        
        
        nib.save(first_img, dir_name1 + '/enhanced.nii.gz')
        nib.save(first_gt,dir_name1 + '/atriumSegImgMO.nii.gz')
        
        nib.save(second_img, dir_name2 + '/enhanced.nii.gz')
        nib.save(second_gt,dir_name2 + '/atriumSegImgMO.nii.gz')
        
        
    
a=np.eye(4)       