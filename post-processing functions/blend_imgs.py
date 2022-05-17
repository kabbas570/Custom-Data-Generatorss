
import numpy as np
import matplotlib.pyplot as plt
img=np.load(r'C:\My data\cada\task1\valid\img\train_5_28.npy')
gt1=np.load(r'C:\My data\cada\task1\valid\seg_gt\train_5_28_gt.npy')
#gt2=np.load(r'C:\My data\cada\task1\valid\b_gt\train_28_23b_gt.npy')
gt3=np.load(r'C:\My data\cada\task1\valid\sc_gt\train_5_28sc_gt.npy')


import nibabel as nib
nii_sc = nib.load(r'C:\Users\Abbas Khan\Downloads\dataset\task1\train_data\train_10\atriumSegImgMO.nii.gz').get_fdata()
nii_sc1 = nib.load(r'C:\Users\Abbas Khan\Downloads\dataset\task1\train_data\train_10\scarSegImgM.nii.gz').get_fdata()

g_seg=nii_sc[:,:,19]
g_sc=nii_sc1[:,:,19]
img_o=nii_sc[:,:,12]



m=img.max()
img=(img-img.min())/img.max()


img=(img-img.mean())/img.std()

gt1[np.where(gt1!=0)]=1
#gt2[np.where(gt2!=0)]=1
gt3[np.where(gt3!=0)]=1

def blend(img,gt1,gt2,gt3):


    img[np.where(gt1==1)]=.6
    img[np.where(gt2==1)]=0.0
    img[np.where(gt3==1)]=0.9
    return img

#blend_=blend(img,gt1,gt2,gt3)


# def blend(image1, image2, ratio):
#     assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

#     alpha = ratio
#     beta = 1 - alpha

#     #coloring yellow.
#     image2 *=  0.7
#     image = image1 * alpha + image2 * beta
#     return image


blend_=blend(img,gt,0.2)

img1=np.concatenate([img,img], axis=0)

R= np.dstack((img, img, img)) 

plt.figure()
plt.imshow(R[:,:,0])