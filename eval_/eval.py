from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import numpy as np


batch_size=44


val_imgs='/data/scratch/acw676/task2_/test/img/'
val_masks='/data/scratch/acw676/task2_/test/gt/'

from g1 import Data_Loader
val_loader=Data_Loader(val_imgs,val_masks,batch_size)

print(len(val_loader))


LEARNING_RATE = 0
weights_paths='/data/home/acw676/seg_/SWIN_1.pth.tar'
#from pytorch_unet import UNet
#model = UNet()
from Swin_Model import SwinUnet
model = SwinUnet()
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)

def check_accuracy(loader, model, device=DEVICE):
    dice_score1=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, t1,label) in enumerate(loop):
            data = data.to(device=DEVICE,dtype=torch.float)
            t1 = t1.to(device=DEVICE,dtype=torch.float)

            p1=model(data)
            p1 = (p1 > 0.5).float()
           
            dice_score1 += (2 * (t1 * p1).sum()) / (
                (t1 + p1).sum() + 1e-8
            )
    print(f"Dice score for Segmentation of LA: {dice_score1/len(val_loader)}")

#def check_accuracy_pos(loader, model, device=DEVICE):
#    dice_score1=0
#    samples_=0
#    loop = tqdm(loader)
#    model.eval()
#    with torch.no_grad():
#        for batch_idx, (data, t1,label) in enumerate(loop):
#            data = data.to(device=DEVICE,dtype=torch.float)
#            t1 = t1.to(device=DEVICE,dtype=torch.float)
#            if torch.sum(t1)!=0: 
#                samples_=samples_+1
#                p1=model(data)
#                p1 = (p1 > 0.5).float()
#                dice_score1 += (2 * (t1 * p1).sum()) / (
#                  (t1 + p1).sum() + 1e-8
#                  )
#    print(f"Dice score for Segmentation of LA: {dice_score1/samples_}")
#    print(dice_score1)
#    print('non zero samples are')
#    print(samples_)
    
    
    
### save the results here #####
g1='/data/home/acw676/examples/results/gt/'
p1='/data/home/acw676/examples/results/pre/'

def save_predictions_as_imgs(
    loader, model,p1_f=p1,g1_f=g1, device=DEVICE):
    loop = tqdm(loader)
    model.eval()
 
    with torch.no_grad():
            for batch_idx, (data, t1,label) in enumerate(loop):
            
                data = data.to(device=DEVICE,dtype=torch.float)
                t1 = t1.to(device=DEVICE,dtype=torch.float)
                
                p1=model(data)
                
                pre1 = (p1 > 0.5).float()
                
                torchvision.utils.save_image(pre1, f"{p1_f}/{label[0]}_p1.png")            
                torchvision.utils.save_image(t1, f"{g1_f}/{label[0]}_g1.png") 

    
def eval_():
    
    #model.load_from() 
    model.to(device=DEVICE,dtype=torch.float)
    
    #model = UNet().to(device=DEVICE,dtype=torch.float)
    checkpoint = torch.load(weights_paths,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    check_accuracy(val_loader, model, device=DEVICE)

    #save_predictions_as_imgs(val_loader, model, device=DEVICE)
    
if __name__ == "__main__":
    eval_()