from tqdm import tqdm
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Early_Stopping import EarlyStopping
import matplotlib.pyplot as plt



   ### SET DATA PATHS & DATA LOADERS #####
   #######################################
   
batch_size=20
train_imgs='/data/scratch/acw676/task2/train/img/'
train_masks='/data/scratch/acw676/task2/train/gt/'
val_imgs='/data/scratch/acw676/task2/valid/img/'
val_masks='/data/scratch/acw676/task2/valid/gt/'

from g4_p import Data_Loader
train_loader=Data_Loader(train_imgs,train_masks,batch_size)
val_loader=Data_Loader(val_imgs,val_masks,batch_size)

print(len(train_loader))
print(len(val_loader))


   ### LOAD MODELS #####
   #######################################

from w_att import m_unet


avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_DS = []  # all training epochs


NUM_EPOCHS=200
LEARNING_RATE=0.0001

def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1,loss_fn2,loss_fn3, scaler):
    
     
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch

    loop = tqdm(loader_train)
    for batch_idx, (img1,img2,img3,gt1,gt2,gt3,label) in enumerate(loop):
        img1 = img1.to(device=DEVICE,dtype=torch.float)
        img2 = img2.to(device=DEVICE,dtype=torch.float)
        img3 = img3.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
        gt2 = gt2.to(device=DEVICE,dtype=torch.float)
        gt3 = gt3.to(device=DEVICE,dtype=torch.float)
        # forward
        with torch.cuda.amp.autocast():
            out1,out2,out3 = model(img1,img2,img3)   
            loss1 = loss_fn1(out1, gt1)
            loss2 = loss_fn2(out2, gt2)
            loss3 = loss_fn3(out3, gt3)
            
        # backward
        loss=0.5*loss1+0.125*loss2+0.125*loss3
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (img1,img2,img3,gt1,gt2,gt3,label) in enumerate(loop_v):
        img1 = img1.to(device=DEVICE,dtype=torch.float)
        img2 = img2.to(device=DEVICE,dtype=torch.float)
        img3 = img3.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
        gt2 = gt2.to(device=DEVICE,dtype=torch.float)
        gt3 = gt3.to(device=DEVICE,dtype=torch.float)
        
        # forward
        with torch.no_grad():
            out1,out2,out3 = model(img1,img2,img3)   
            loss1 = loss_fn1(out1, gt1)
            loss2 = loss_fn2(out2, gt2)
            loss3 = loss_fn3(out3, gt3)
        
        loss=0.5*loss1+0.125*loss2+0.125*loss3
        loop_v.set_postfix(loss=loss.item())
        valid_losses.append(loss.item())
    model.train()
    
    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    

def save_checkpoint(state, filename="w_att_1.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def check_accuracy(loader, model, device=DEVICE):
    dice_score1=0
    dice_score2=0
    dice_score3=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,img2,img3,gt1,gt2,gt3,label) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            img2 = img2.to(device=DEVICE,dtype=torch.float)
            img3 = img3.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            gt2 = gt2.to(device=DEVICE,dtype=torch.float)
            gt3 = gt3.to(device=DEVICE,dtype=torch.float)
           
            p1,p2,p3 = model(img1,img2,img3)  
            
            p1 = (p1 > 0.5).float()
            dice_score1 += (2 * (p1 * gt1).sum()) / (
                (p1 + gt1).sum() + 1e-8)
            
            p2 = (p2 > 0.5).float()
            dice_score2 += (2 * (p2 * gt2).sum()) / (
                (p2 + gt2).sum() + 1e-8)
            
            p3 = (p3 > 0.5).float()
            dice_score3 += (2 * (p3 * gt3).sum()) / (
                (p3 + gt3).sum() + 1e-8)
           
    print(f"Dice score: {dice_score1/len(loader)}")
    print(f"Dice score: {dice_score2/len(loader)}")
    print(f"Dice score: {dice_score3/len(loader)}")

    model.train()
    return dice_score1/len(loader)
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection   
        IoU = (intersection + smooth)/(union + smooth)          
        return 1 - IoU
 
epoch_len = len(str(NUM_EPOCHS))
early_stopping = EarlyStopping(patience=10, verbose=True)
                      
def main():
    model = m_unet().to(device=DEVICE,dtype=torch.float)
    loss_fn1 =IoULoss()
    loss_fn2 =IoULoss()
    loss_fn3 =IoULoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,loss_fn2,loss_fn3,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{NUM_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        dice_score= check_accuracy(val_loader, model, device=DEVICE)
        avg_valid_DS.append(dice_score.detach().cpu().numpy())
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            break

if __name__ == "__main__":
    main()


avg_train_losses=avg_train_losses
avg_train_losses=avg_train_losses
avg_valid_DS=avg_valid_DS
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
plt.plot(range(1,len(avg_valid_DS)+1),avg_valid_DS,label='Validation DS')

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
fig.savefig('w_att_1.png', bbox_inches='tight')   
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
