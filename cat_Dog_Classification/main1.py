   #### Specify all the paths here #####
train_imgs = '/data/scratch/acw676/mahmud/train/'
val_imgs = '/data/scratch/acw676/mahmud/train/'
path_to_save_check_points = 'C:\My_Data\mahmud'+'/VGG16_Cat_Dog'
path_to_save_Learning_Curve = 'C:\My_Data\mahmud'+'/VGG16_Cat_Dog'

        #### Specify all the Hyperparameters\image dimenssions here #####
batch_size = 2
Max_Epochs = 100
LEARNING_RATE=0.0001
Patience = 5

        #### Import All libraies used for training  #####
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import cv2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from Early_Stopping import EarlyStopping

### Data_Generators ########

from Data_Generator import Data_Loader
train_loader = Data_Loader(train_imgs,batch_size)
val_loader = Data_Loader(val_imgs,batch_size)
   ### Load the Data using Data generators and paths specified #####
   #######################################
print(len(train_loader)) ### this shoud be = Total_images/ batch size
print(len(val_loader))   ### same here

### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_Acc = []  # all training epochs



from Model import VGG16
model_=VGG16()
### Next we have all the funcitons which will be called in the main for training ####
    
### 2- the main training fucntion to update the weights....
def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1,scaler):
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch
    model.train()
    loop = tqdm(loader_train)
    for batch_idx, (image,label) in enumerate(loop):
        image = image.to(device=DEVICE,dtype=torch.float)  
        label = label.to(device=DEVICE,dtype=torch.float)
        
        with torch.cuda.amp.autocast():
            out= model(image)   
            loss = loss_fn1(out, label)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses.append(float(loss))
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (image,label) in enumerate(loop):
        image = image.to(device=DEVICE,dtype=torch.float)  
        label = label.to(device=DEVICE,dtype=torch.float)

        with torch.no_grad():
            
            out= model(image)   
            loss = loss_fn1(out, label)
            
        # backward
        loop_v.set_postfix(loss = loss.item())
        valid_losses.append(float(loss))

    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

PREDS=[]
TGS=[]

def check_accuracy(loader, model, device=DEVICE):
    correct=0
    total_instances=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,label) in enumerate(loop):
            data = data.to(device=DEVICE,dtype=torch.float)
            label = label.to(device=DEVICE,dtype=torch.long)

            p1=model(data)
            p1 = (p1 > 0.5)* 1
            correct_batch=torch.sum(p1 == label)
            correct+=correct_batch
            total_instances += p1.shape[0]
            PREDS.append(p1)
            TGS.append(label)

    acc= correct / total_instances
    print('\n accuray is : \n', acc)
    return acc


BCE = nn.BCELoss()

## 7- This is the main Training function, where we will call all previous functions
       
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=Patience, verbose=True)

def main():
    model = model_.to(device=DEVICE,dtype=torch.float)

    loss_fn =BCE
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
    # optimizer = optim.SGD(model.parameters(), momentum=0.9 ,lr=LEARNING_RATE)  ### SGD
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        train_loss,valid_loss = train_fn(train_loader,val_loader, model, optimizer, loss_fn,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        Acc_= check_accuracy(val_loader, model, device=DEVICE)
        avg_valid_Acc.append(Acc_.detach().cpu().numpy())
        
        early_stopping(valid_loss, Acc_)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            break

if __name__ == "__main__":
    main()

### This part of the code will generate the learning curve ......

avg_train_losses=avg_train_losses
avg_train_losses=avg_train_losses
avg_valid_Acc=avg_valid_Acc
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
plt.plot(range(1,len(avg_valid_Acc)+1),avg_valid_Acc,label='Validation DS')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')