save_pre_path='/data/scratch/acw676/patch_2/filtered/results/'
def blend(image1,gt,pre, ratio=0.5):
    
    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    alpha = ratio
    beta = 1 - alpha
    theta=beta-0.1

    #coloring yellow.
    gt *= [0.2,0.7, 0] ### Green Color
    pre*=[1,0,0]   ## Red Color
    image = image1 * alpha + gt * beta+ pre * theta
    return image

def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
def save_predictions_as_imgs(
    loader, model, device=DEVICE):
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,gt1,label) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            
            p1= model(img1)              
            p1 = (p1 > 0.5).float()   
    
            for k in range(img1.shape[0]):
                img=img1[k,0,:,:]
                t1=gt1[k,0,:,:]
                pre1=p1[k,0,:,:]
                name=label[k]
                
                img=img.cpu().numpy()
                t1=t1.cpu().numpy()
                pre1=pre1.cpu().numpy()
                                
                img=normalize(img)
                stacked_img = np.stack((img,)*3, axis=-1)
                stacked_gt = np.stack((pre1,)*3, axis=-1)
                stacked_pre = np.stack((t1,)*3, axis=-1)
                
                result=blend(stacked_img,stacked_gt,stacked_pre)
                
                plt.imsave(os.path.join(save_pre_path,name+".png"),result)
