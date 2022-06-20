import glob
import numpy as np
import cv2
import os

img_id = []
for infile in sorted(glob.glob(r'C:\My data\sateg0\train\img\*.npy')):
    img_id.append(infile)


def proc_(img,gt_LA,gt_sc):
    kernel = np.ones((5,5),np.uint8)
    
    dilation = cv2.dilate(gt_LA,kernel,iterations = 5)
    erosion = cv2.erode(gt_LA,kernel,iterations = 5)
    diff1=dilation-gt_LA
    diff2=gt_LA-erosion
    
    
    gen=np.zeros([640,640])
    gen[np.where(diff1==1)]=1
    gen[np.where(diff2==1)]=1
    
    new_img=img*gen
    
    
    x=[]
    y=[]
    row, col = gt_LA.shape
    for i in range(row):
        for j in range(col):
            if new_img[i,j] != 0:    
                x.append(j) # get x indices
                y.append(i) # get y indices
    
    if len(x)==0:
        min_x=221
        max_x=333
        min_y=198
        max_y=361
        cropped_area =new_img[min_y:max_y,min_x:max_x]
        
        cropped_gt =gt_sc[min_y:max_y,min_x:max_x]
    else : 
        min_x=min(x)
        max_x=max(x)
        
        min_y=min(y)
        max_y=max(y)
        
        cropped_area =new_img[min_y:max_y,min_x:max_x]
        
        cropped_gt =gt_sc[min_y:max_y,min_x:max_x]
        
        cropped_gt[np.where(cropped_gt!=0)]=1
    
    return cropped_area,cropped_gt,min_x,max_x,min_y,max_y


img_path=r'C:/Users/Abbas Khan/Desktop/PROS/nw_data/train/img/'
gt_path=r'C:/Users/Abbas Khan/Desktop/PROS/nw_data/train/gt/'

img_path = os.path.join(img_path)
gt_path = os.path.join(gt_path)



gt_apth=r'C:\My data\sateg0\valid\seg_gt'
sc_path=r'C:\My data\sateg0\valid\sc_gt'
for k in range(528):
    name=img_id[k][28:-4]
    #print(name)
    img=np.load(img_id[k])
    
    gt_id=gt_apth+'/'+name+'_gt'+'.npy'
    gt_LA=np.load(gt_id)
    
    sc_id=sc_path+'/'+name+'sc_gt'+'.npy'
    gt_sc=np.load(sc_id)
    
    
    
    
    if img.shape[0]==576:
          temp=np.zeros([640,640])
          temp1=np.zeros([640,640])
          temp2=np.zeros([640,640])
          
          temp[32:608, 32:608] = img
          img=temp
          
          temp1[32:608, 32:608] = gt_LA
          gt_LA=temp1
          
          temp2[32:608, 32:608] = gt_sc
          gt_sc=temp2
    
    img_,gt_,min_x,max_x,min_y,max_y=proc_(img,gt_LA,gt_sc)
    
    ##save new data
    
    np.save(img_path+name,img_)
    np.save(gt_path+name+'_gt',gt_)
    
    



# g=np.load(r'C:\Users\Abbas Khan\Desktop\PROS\nw_data\gt\train_7_31_gt.npy')
# i=np.load(r'C:\Users\Abbas Khan\Desktop\PROS\nw_data\img\train_7_31.npy')





# o1=np.load(r'C:\My data\sateg0\valid\sc_gt\train_7_31sc_gt.npy')
# o2=np.load(r'C:\My data\sateg0\valid\img\train_7_31.npy')

# o3=np.load(r'C:\My data\sateg0\valid\seg_gt\train_7_31_gt.npy')





