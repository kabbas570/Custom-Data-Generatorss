import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import cv2
from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
from sklearn.metrics import confusion_matrix

batch_size=1

test_data=r'/data/scratch/acw676/testing/task1_both/'

from g_task1 import Data_Loader
test_loader=Data_Loader(test_data,batch_size)



from models_task1 import Model_LA, Model_Scar
model_la = Model_LA()
model_sc = Model_Scar()

LEARNING_RATE = 0
weights_paths_la="/data/home/acw676/task1_val/m_unet4_Final.pth.tar"
weights_paths_scar="/data/home/acw676/task1_val/Model_2.pth.tar"

optimizer_la = optim.Adam(model_la.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
optimizer_scar = optim.Adam(model_sc.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)

def Evaluation_Metrics(pre,gt):
    pre=pre.flatten() 
    gt=gt.flatten()  
    tn, fp, fn, tp=confusion_matrix(gt,pre,labels=[0,1]).ravel()
    
    iou=tp/(tp+fn+fp) 
    dice=2*tp/(2*tp + fp + fn)
    return iou,dice,tp,tn,fp,fn 

save_pre_path='/data/home/acw676/temp/'  
savedir  ='/data/home/acw676/temp/'   

 
def check_accuracy(test_loader, model_la,model_sc, device=DEVICE):
    loop = tqdm(test_loader)
    model_la.eval()
    model_sc.eval()
    with torch.no_grad():
        for batch_idx, (img1,img2,gt1,gt2,label,sp_dim) in enumerate(loop):
        
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            img2 = img2.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            gt2 = gt2.to(device=DEVICE,dtype=torch.float)
           
            img1=torch.moveaxis(img1, [0,1,2,3],[1,0,2,3])
            img2=torch.moveaxis(img2, [0,1,2,3],[1,0,2,3])
            gt1=torch.moveaxis(gt1, [0,1,2,3],[1,0,2,3])
            gt2=torch.moveaxis(gt2, [0,1,2,3],[1,0,2,3])
            
            
            pred1=[]

            for i in range(img1.shape[0]):
              
              single_img1=img1[i,:,:,:]
              single_img2=img2[i,:,:,:]
              
              single_img1=torch.unsqueeze(single_img1, axis=0)
              single_img2=torch.unsqueeze(single_img2, axis=0)

              p1 = model_la(single_img1,single_img2)  
            
              p1 = (p1 > 0.5) * 1        
              
              p1=p1.cuda().cpu()
              pred1.append(p1)
              
            pred2 = [item.numpy() for item in pred1]
            pred2 = np.array(pred2)
            pred2=pred2[:,0,:,:,:]
            

            pred2=pred2[:,0,:,:]

            if sp_dim[0]==576:
              print(sp_dim[0])
              pred2=pred2[:,32:608,32:608]
            
            
            pred2=np.transpose(pred2, (1,2,0))
            
            ## saving .nni files predictions ####
            to_format_pre = nib.Nifti1Image(pred2, np.eye(4))  
            name='LA_predict'+ label[0]
            to_format_pre.to_filename(os.path.join(savedir,name+'.nii.gz'))  # Save as NiBabel file
            
                      ########  scrar predictions ###
                      ### extarcting the boundary ####
                      
            walls=[]
            for j in range(pred2.shape[2]):
                la_pre=pred2[:,:,j]
                la_pre = la_pre.astype(np.uint8)
                contours, hierarchy = cv2.findContours(image=la_pre, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                # # draw contours on the original image
                la_pre_copy = la_pre.copy()
                cv2.drawContours(image=la_pre_copy, contours=contours, contourIdx=-1, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
                wall=np.zeros([la_pre.shape[0],la_pre.shape[1]])
                wall[np.where(la_pre_copy>=200)]=1
                
                
                walls.append(wall)
            #walls = [item.numpy() for item in walls]
            walls = np.array(walls)
            walls=np.transpose(walls, (1,2,0))
            print(walls.shape)
            ## saving .nni files predictions ####
            to_format_pre = nib.Nifti1Image(walls, np.eye(4))  
            name='Wall_predict'+ label[0]
            to_format_pre.to_filename(os.path.join(savedir,name+'.nii.gz'))  # Save as NiBabel file
            
            #### now predict scars from wall ###
  

    
def eval_():
    
    ##LA Model ###
    model_la.to(device=DEVICE,dtype=torch.float)
    checkpoint = torch.load(weights_paths_la,map_location=DEVICE)
    model_la.load_state_dict(checkpoint['state_dict'])
    optimizer_la.load_state_dict(checkpoint['optimizer'])
    
    #### Scar Model ###
    model_sc.to(device=DEVICE,dtype=torch.float)
    checkpoint = torch.load(weights_paths_scar,map_location=DEVICE)
    model_sc.load_state_dict(checkpoint['state_dict'])
    optimizer_scar.load_state_dict(checkpoint['optimizer'])
    

    check_accuracy(test_loader,model_la, model_sc, device=DEVICE)

    #save_predictions_as_imgs(test_loader, model, device=DEVICE)
    
if __name__ == "__main__":
    eval_()
    



