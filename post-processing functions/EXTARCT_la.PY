import glob
import numpy as np
import cv2
import os

img_id = []
for infile in sorted(glob.glob(r'C:\My data\sateg0\train\img\*.npy')):
    img_id.append(infile)


img_path=r'C:/Users/Abbas Khan/Desktop/PROS/nw_data/train1/img/'
gt_path=r'C:/Users/Abbas Khan/Desktop/PROS/nw_data/train1/gt/'

img_path = os.path.join(img_path)
gt_path = os.path.join(gt_path)


def proc_(img,gt_LA,name):
    
    name=name
    kernel = np.ones((5,5),np.uint8)
    
    dilation = cv2.dilate(gt_LA,kernel,iterations = 5)
    
    gen=np.zeros([640,640])
    gen[np.where(dilation==1)]=1
    
    new_img=img*gen
    
    
    x=[]
    y=[]
    row, col = gt_LA.shape
    for i in range(row):
        for j in range(col):
            if new_img[i,j] != 0:    
                x.append(j) # get x indices
                y.append(i) # get y indices
    
    if len(x)!=0:

        min_x=min(x)
        max_x=max(x)
        
        min_y=min(y)
        max_y=max(y)
        
        cropped_area =new_img[min_y:max_y,min_x:max_x]
        
        cropped_gt =gt_LA[min_y:max_y,min_x:max_x]
        
        cropped_gt[np.where(cropped_gt!=0)]=1
        
        if np.sum(cropped_gt)!=0:
            np.save(img_path+name,cropped_area)
            np.save(gt_path+name+'_gt',cropped_gt)
        
        
    
        #return cropped_area,cropped_gt,min_x,max_x,min_y,max_y


# img_path=r'C:/Users/Abbas Khan/Desktop/PROS/nw_data/train/img/'
# gt_path=r'C:/Users/Abbas Khan/Desktop/PROS/nw_data/train/gt/'

# img_path = os.path.join(img_path)
# gt_path = os.path.join(gt_path)



gt_apth=r'C:\My data\sateg0\train\seg_gt'
for k in range(2112):
    name=img_id[k][28:-4]
    #print(name)
    img=np.load(img_id[k])
    
    gt_id=gt_apth+'/'+name+'_gt'+'.npy'
    gt_LA=np.load(gt_id)
    
    
    if img.shape[0]==576:
          temp=np.zeros([640,640])
          temp1=np.zeros([640,640])
          temp2=np.zeros([640,640])
          
          temp[32:608, 32:608] = img
          img=temp
          
          temp1[32:608, 32:608] = gt_LA
          gt_LA=temp1
         
    #img_,gt_,min_x,max_x,min_y,max_y=proc_(img,gt_LA,gt_sc)
    
    _=proc_(img,gt_LA,name)
    
    ##save new data
    
    # np.save(img_path+name,img_)
    # np.save(gt_path+name+'_gt',gt_)
