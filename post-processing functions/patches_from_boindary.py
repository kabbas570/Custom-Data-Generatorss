import torch

def make_patches(wall,single_img,gt_sc):  ##wall---->[640,640],  img--->[640,640]
      
    img_batch=[]
    gt_batch=[]
    for x in range(wall.shape[0]):
        for y in range(wall.shape[1]):
            if wall[y,x]==1.0:
                img_crop=single_img[y-32:y+32,x-32:x+32]
                
                gt_sc_cropped=gt_sc[y-32:y+32,x-32:x+32]
                    
                #img_crop=img_crop.cpu()
                #img_crop=img_crop.numpy()
                            
                # mean=np.mean(img_crop,keepdims=True)
                # std=np.std(img_crop,keepdims=True)
                # img_crop=(img_crop-mean)/std
                            
                gen_img=np.zeros([3,img_crop.shape[0],img_crop.shape[1]]) 
                    
                img_o=img_crop.copy()    ## orignal img
                    
                img_crop[np.where(img_crop<0)]=0   #### no negatives
                    
                hitss=np.histogram(img_crop, bins=5)
                    
                value_=hitss[1][1]
                img1=np.zeros([img_crop.shape[0],img_crop.shape[1]])  
                img1[np.where(img_crop>=value_)]=1  
                new_img=img1*img_o                 #### from hists
                    
                gen_img[0,:,:]=img_o
                gen_img[1,:,:]=img_crop
                gen_img[2,:,:]=new_img
                        
                #gen_img=torch.tensor(gen_img,dtype=torch.float)
                #gen_img=torch.unsqueeze(gen_img, 0)
                
                img_batch.append(gen_img)
                gt_batch.append(gt_sc_cropped)
    
    img_batch=np.array(img_batch)
    gt_batch=np.array(gt_batch)
    
    return img_batch,gt_batch
    
    

img_batch,gt_batch=make_patches(wall,single_img,single_sc)
