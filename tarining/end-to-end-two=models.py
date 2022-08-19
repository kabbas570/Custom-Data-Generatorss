from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from Early_Stopping import EarlyStopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import matplotlib.pyplot as plt
import cv2

   ### LOAD MODELS #####
   #######################################
   
from Models import m_unet4,m_unet_33_1


   ### SET DATA PATHS & DATA LOADERS #####
   #######################################

from g1 import Data_Loader

image_path_train = '/data/scratch/acw676/new_data/task1/train/img/'
mask_path_train = '/data/scratch/acw676/new_data/task1/train/seg_gt/'
sc_path_train='/data/scratch/acw676/new_data/task1/train/sc_gt/'

image_path_val = '/data/scratch/acw676/new_data/task1/valid/img/'
mask_path_val = '/data/scratch/acw676/new_data/task1/valid/seg_gt/'
sc_path_val='/data/scratch/acw676/new_data/task1/valid/sc_gt/'


batch_size=4
val_loader=Data_Loader(image_path_val,mask_path_val,sc_path_val,batch_size)
train_loader=Data_Loader(image_path_train,mask_path_train,sc_path_train,batch_size)

print(len(train_loader))
print(len(val_loader))


avg_train_losses1 = []   # losses of all training epochs
avg_valid_losses1 = []  #losses of all training epochs
avg_valid_DS1 = []  # all training epochs

avg_train_losses2 = []   # losses of all training epochs
avg_valid_losses2 = []  #losses of all training epochs
avg_valid_DS2 = []  # all training epochs

NUM_EPOCHS=200
LEARNING_RATE=0.0001

def make_patches(single_img,gt_sc):  ##wall---->[640,640],  img--->[640,640]  
    img_batch=[]
    gt_batch=[]
    for x in range(gt_sc.shape[0]):
        for y in range(gt_sc.shape[1]):
            if gt_sc[y,x]==1.0:
                img_crop=single_img[y-32:y+32,x-32:x+32]
                
                gt_sc_cropped=gt_sc[y-32:y+32,x-32:x+32]
                gt_sc_cropped=np.expand_dims(gt_sc_cropped,0)
                
                #img_crop=img_crop.cpu()
                #img_crop=img_crop.numpy()
                            
                mean=np.mean(img_crop,keepdims=True)
                std=np.std(img_crop,keepdims=True)
                img_crop=(img_crop-mean)/std
                            
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


def train_fn(loader_train,loader_valid, model1,model2,optimizer1,optimizer2,loss_fn1,scaler): 
     
    train_losses1 = [] # loss of each batch
    valid_losses1 = []  # loss of each batch
    
    train_losses2 = [] # loss of each batch
    valid_losses2 = []  # loss of each batch
    
    model1.train()
    model2.train()

    loop = tqdm(loader_train)
    for batch_idx, (img,gt1,gt2,label) in enumerate(loop):
        img = img.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
        gt2 = gt2.to(device=DEVICE,dtype=torch.float)
    
        # forward for model1
        with torch.cuda.amp.autocast():
            out1 = model1(img)   
            loss1 = loss_fn1(out1, gt1)
        # backward
        optimizer1.zero_grad()
        scaler.scale(loss1).backward()
        scaler.step(optimizer1)
        scaler.update()
        # update tqdm loop
        #loop.set_postfix(loss=loss1.item())
        
        loss1_ = float(loss1) 
        train_losses1.append(loss1_)
        
        ### make patches from out1 and feed to model2 #########
        
#        out1=out1.cpu()
#        out1=out1.numpy()
#        out1 = out1.astype(np.uint8)
        
        loss_patch_epoch=0
        for k in range(out1.shape[0]):
          #single_la=out1[k,0,:,:]    ##-->[640,640]
          single_img=img[k,0,:,:]   ##-->[640,640]
          single_sc_gt=gt2[k,0,:,:]  ##-->[640,640]
          
          # contours, hierarchy = cv2.findContours(image=single_la, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
          # single_la_copy = single_la.copy()
          # cv2.drawContours(image=single_la_copy, contours=contours, contourIdx=-1, color=(255,0,0), thickness=1, lineType=cv2    .LINE_AA)      
          # wall=np.zeros([single_la.shape[0],single_la.shape[1]])         
          # wall[np.where(single_la_copy>=200)]=1   ### ---> this wall has shape of [640,640]
          
          
          single_img=single_img.cpu().numpy()
          single_sc_gt=single_sc_gt.cpu().numpy()
          
          if np.sum(single_sc_gt)!=0:
              #print(np.sum(single_sc_gt))
              img_batch,gt_batch=make_patches(single_img,single_sc_gt)
              
              img_batch=torch.tensor(img_batch)
              gt_batch=torch.tensor(gt_batch)
              img_batch = img_batch.to(device=DEVICE,dtype=torch.float)
              gt_batch = gt_batch.to(device=DEVICE,dtype=torch.float)
              
              
              #print(img_batch.shape)
              #print(gt_batch.shape)
                                                    
                    # forward for model2
              with torch.cuda.amp.autocast():
                        out2 = model2(img_batch) 
                        #print(out2.shape)  
                        loss2 = loss_fn1(out2, gt_batch)
               
                    # backward
              loss_patch_epoch=loss_patch_epoch+float(loss2)
              optimizer2.zero_grad()
              scaler.scale(loss2).backward()
              scaler.step(optimizer2)
              scaler.update()
                # update tqdm loop
                
        loop.set_postfix(loss=loss1.item())
        train_losses2.append(loss_patch_epoch)
                

    loop_v = tqdm(loader_valid)
    model1.eval()
    model2.eval()
    for batch_idx, (img,gt1,gt2,label) in enumerate(loop_v):
        img = img.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
        gt2 = gt2.to(device=DEVICE,dtype=torch.float)
        
        # forward
        with torch.no_grad():
            out1 = model1(img)   
            loss1 = loss_fn1(out1, gt1)
        
        loss1_ = float(loss1) 
        #loop_v.set_postfix(loss=loss1.item())
        valid_losses1.append(loss1_)
        
        loss_patch_epoch=0
        for q in range(out1.shape[0]):
          #single_la=out1[k,0,:,:]    ##-->[640,640]
          single_img=img[q,0,:,:]   ##-->[640,640]
          single_sc_gt=gt2[q,0,:,:]  ##-->[640,640]
          
          single_img=single_img.cpu().numpy()
          single_sc_gt=single_sc_gt.cpu().numpy()
          
          # contours, hierarchy = cv2.findContours(image=single_la, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
          # single_la_copy = single_la.copy()
          # cv2.drawContours(image=single_la_copy, contours=contours, contourIdx=-1, color=(255,0,0), thickness=1, lineType=cv2    .LINE_AA)      
          # wall=np.zeros([single_la.shape[0],single_la.shape[1]])         
          # wall[np.where(single_la_copy>=200)]=1   ### ---> this wall has shape of [640,640]
          
          #print('I am here')
          if np.sum(single_sc_gt)!=0:
              #print(np.sum(single_sc_gt))
              
              img_batch,gt_batch=make_patches(single_img,single_sc_gt)
              
              img_batch=torch.tensor(img_batch)
              gt_batch=torch.tensor(gt_batch)
              img_batch = img_batch.to(device=DEVICE,dtype=torch.float)
              gt_batch = gt_batch.to(device=DEVICE,dtype=torch.float)
                                                    
                    # forward for model2
              with torch.cuda.amp.autocast():
                        out2 = model2(img_batch)   
                        loss2 = loss_fn1(out2, gt_batch)
               
                    # backward
              loss_patch_epoch=loss_patch_epoch+float(loss2)
        loop_v.set_postfix(loss=loss1.item())
        valid_losses2.append(loss_patch_epoch)
        
        
    #model1.train()
    #model2.train()
    
    train_loss_per_epoch1 = np.average(train_losses1)
    valid_loss_per_epoch1 = np.average(valid_losses1)
    ## all epochs
    avg_train_losses1.append(train_loss_per_epoch1)
    avg_valid_losses1.append(valid_loss_per_epoch1)
    
    
    train_loss_per_epoch2 = np.average(train_losses2)
    valid_loss_per_epoch2 = np.average(valid_losses2)
    ## all epochs
    avg_train_losses2.append(train_loss_per_epoch2)
    avg_valid_losses2.append(valid_loss_per_epoch2)
    
    return train_loss_per_epoch1,valid_loss_per_epoch1,train_loss_per_epoch2,valid_loss_per_epoch2
    


def save_checkpoint1(state, filename="LA_Weights.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def save_checkpoint2(state, filename="Scar_Weights.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def check_accuracy(loader, model1,model2, device=DEVICE):
    dice_score1=0
    dice_score2=0
    patch_num=0
    loop = tqdm(loader)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for batch_idx, (img,gt1,gt2,label) in enumerate(loop):
            img = img.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            gt2 = gt2.to(device=DEVICE,dtype=torch.float)
            
            
            with torch.cuda.amp.autocast():
                p1 = model1(img)  
            p1 = (p1 > 0.5) * 1
            dice_score1 += (2 * (p1 * gt1).sum()) / (
                (p1 + gt1).sum() + 1e-8)
            
            for k in range(p1.shape[0]):
              #single_la=out1[k,0,:,:]    ##-->[640,640]
              single_img=img[k,0,:,:]   ##-->[640,640]
              single_sc_gt=gt2[k,0,:,:]  ##-->[640,640]
              
              
              single_img=single_img.cpu().numpy()
              single_sc_gt=single_sc_gt.cpu().numpy()
              
              # contours, hierarchy = cv2.findContours(image=single_la, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
              # single_la_copy = single_la.copy()
              # cv2.drawContours(image=single_la_copy, contours=contours, contourIdx=-1, color=(255,0,0), thickness=1, lineType=cv2    .LINE_AA)      
              # wall=np.zeros([single_la.shape[0],single_la.shape[1]])         
              # wall[np.where(single_la_copy>=200)]=1   ### ---> this wall has shape of [640,640]
              
              if np.sum(single_sc_gt)!=0:
                  #print(np.sum(single_sc_gt))
                  patch_num=patch_num+1
                  img_batch,gt_batch=make_patches(single_img,single_sc_gt)
                  
                  img_batch=torch.tensor(img_batch)
                  gt_batch=torch.tensor(gt_batch)
                  img_batch = img_batch.to(device=DEVICE,dtype=torch.float)
                  gt_batch = gt_batch.to(device=DEVICE,dtype=torch.float)
                  
                        # forward for model2
                  with torch.cuda.amp.autocast():
                            p2 = model2(img_batch)  
                
                  p2 = (p2 > 0.5) * 1
                  dice_score2 += (2 * (p2 * gt_batch).sum()) / (
                     (p2 + gt_batch).sum() + 1e-8)
            
                
    print(f"Dice score for LA: {dice_score1/len(loader)}")
    print(f"Dice score for Scars: {dice_score2/patch_num}")

    #model1.train()
    #model2.train()
    return dice_score1/len(loader),dice_score2/len(loader)
    
    
             
ALPHA = 0.3
BETA = 0.7
GAMMA = .75

class FocalTverskyLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self)._init_()

    def forward(self, inputs, targets, smooth=.0001, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
        
epoch_len = len(str(NUM_EPOCHS))
early_stopping = EarlyStopping(patience=5, verbose=True)
                      
def main():
    model1 = m_unet4().to(device=DEVICE,dtype=torch.float)
    model2 = m_unet_33_1().to(device=DEVICE,dtype=torch.float)
    loss_fn1 =FocalTverskyLoss()

    optimizer1 = optim.Adam(model1.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_loss1,valid_loss1, train_loss2,valid_loss2=train_fn(train_loader,val_loader, model1,model2, optimizer1,optimizer2, loss_fn1,scaler)
        
        train_loss=train_loss1+train_loss2
        valid_loss=valid_loss1+valid_loss2
        print_msg = (f'[{epoch:>{epoch_len}}/{NUM_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint1 = {
            "state_dict": model1.state_dict(),
            "optimizer":optimizer1.state_dict(),
        }
        save_checkpoint1(checkpoint1)
        
        checkpoint2 = {
            "state_dict": model2.state_dict(),
            "optimizer":optimizer2.state_dict(),
        }
        save_checkpoint2(checkpoint2)
        
        
        dice_score1,dice_score2= check_accuracy(val_loader, model1,model2, device=DEVICE)
        
        avg_valid_DS1.append(dice_score1.detach().cpu().numpy())
        avg_valid_DS2.append(dice_score2.detach().cpu().numpy())
        
        dice_score=dice_score1+dice_score2
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            break

if __name__ == "__main__":
    main()



# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses1)+1),avg_train_losses1, label='Training Loss LA Model')
plt.plot(range(1,len(avg_valid_losses1)+1),avg_valid_losses1,label='Validation Loss LA Model')
plt.plot(range(1,len(avg_valid_DS1)+1),avg_valid_DS1,label='Validation DS LA Model')

plt.plot(range(1,len(avg_train_losses2)+1),avg_train_losses2, label='Training Loss Scar Model')
plt.plot(range(1,len(avg_valid_losses2)+1),avg_valid_losses2,label='Validation Loss car Model')
plt.plot(range(1,len(avg_valid_DS2)+1),avg_valid_DS2,label='Validation DS car Model')


# find position of lowest validation loss
minposs = avg_valid_losses1.index(min(avg_valid_losses1))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses1)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('end_to_end1.png', bbox_inches='tight')   
