from Swin_Model import SwinUnet
from tqdm import tqdm
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Early_Stopping import EarlyStopping
import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss

batch_size=32
train_imgs='/data/scratch/acw676/task2/train/img/'
train_masks='/data/scratch/acw676/task2/train/gt/'
val_imgs='/data/scratch/acw676/task2/valid/img/'
val_masks='/data/scratch/acw676/task2/valid/gt/'


#train_imgs='/data/scratch/acw676/task2_/test/img/'
#train_masks='/data/scratch/acw676/task2_/test/gt/'
#val_imgs='/data/scratch/acw676/task2_/test/img/'
#val_masks='/data/scratch/acw676/task2_/test/gt/'


from g1 import Data_Loader
train_loader=Data_Loader(train_imgs,train_masks,batch_size)

val_loader=Data_Loader(val_imgs,val_masks,batch_size)


print(len(train_loader))
print(len(val_loader))


avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_DS = []  # all training epochs


NUM_EPOCHS=200
LEARNING_RATE=0.001

def train_fn(loader_train,loader_valid, model, optimizer, loss_fn1,loss_fn2, scaler):
    
     
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch

    
    loop = tqdm(loader_train)
    for batch_idx, (data, t1,label) in enumerate(loop):
        data = data.to(device=DEVICE,dtype=torch.float)
        t1 = t1.to(device=DEVICE,dtype=torch.float)
        # forward
        with torch.cuda.amp.autocast():
            out1 = model(data)    
            loss_ce = loss_fn1(out1, t1[:])
            loss_dice = loss_fn2(out1, t1)
        loss = 0.4 * loss_ce + 0.6 * loss_dice
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (data, t1,label) in enumerate(loop_v):
        data = data.to(device=DEVICE,dtype=torch.float)
        t1 = t1.to(device=DEVICE,dtype=torch.float)
        # forward
        with torch.cuda.amp.autocast():
            out1 = model(data)    
            loss_ce = loss_fn1(out1, t1[:])
            loss_dice = loss_fn2(out1, t1)
        loss = 0.4 * loss_ce + 0.6 * loss_dice
        loop_v.set_postfix(loss=loss.item())
        valid_losses.append(loss.item())
    model.train()
    
    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    

def save_checkpoint(state, filename="SWIN_OL.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def check_accuracy(loader, model, device=DEVICE):
    dice_score1=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,t1,label) in enumerate(loop):
            data = data.to(device=DEVICE,dtype=torch.float)
            t1 = t1.to(device=DEVICE,dtype=torch.float)
           
            p1=model(data)
            
            p1 = (p1 > 0.5).float()
            dice_score1 += (2 * (p1 * t1).sum()) / (
                (p1 + t1).sum() + 1e-8)
           
    print(f"Dice score: {dice_score1/len(loader)}")

    model.train()
    return dice_score1/len(loader)
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes=1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        #print(target.shape)
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        #target = self._one_hot_encoder(target)
        #print(target.shape)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

 
epoch_len = len(str(NUM_EPOCHS))
early_stopping = EarlyStopping(patience=5, verbose=True)
                      
def main():
    model = SwinUnet().to(device=DEVICE,dtype=torch.float) 
    model.load_from() 
    #model = model.to(device=DEVICE,dtype=torch.float)    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, ce_loss,dice_loss,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{NUM_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        #save_checkpoint(checkpoint)
        dice_score= check_accuracy(val_loader, model, device=DEVICE)
        avg_valid_DS.append(dice_score.detach().cpu().numpy())
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            save_checkpoint(checkpoint)
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
fig.savefig('SWIN_OL.png', bbox_inches='tight')   
    
    

    




