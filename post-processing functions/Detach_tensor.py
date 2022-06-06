def check_accuracy(loader, model,folder1=g1,folder2=g2,folder3=g3, device=DEVICE):
    dice_score1=0
    dice_score2=0
    dice_score3=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, t1,t2,t3,label) in enumerate(loop):
            data = data.to(device=DEVICE,dtype=torch.float)
            t1 = t1.to(device=DEVICE,dtype=torch.float)
            t2 = t2.to(device=DEVICE,dtype=torch.float)
            t3 = t3.to(device=DEVICE,dtype=torch.float)
            
            p1,p2,p3=model(data)
            
            p1 = (p1 > 0.5).float()
            p2 = (p2 > 0.5).float()
            p3 = (p3 > 0.5).float()


            dice_score1 += (2 * (p1 * t1).sum()) / (
                (p1 + t1).sum() + 1e-8
            )
            dice_score2 += (2 * (p2 * t2).sum()) / (
                (p2 + t2).sum() + 1e-8
            )
            dice_score3 += (2 * (p3 * t3).sum()) / (
                (p3 + t3).sum() + 1e-8
            )
            
            s1=(2 * (p1 * t1).sum()) / (
                (p1 + t1).sum() + 1e-8)  
            print(s1)
            
#            p1=p1.to("cpu").numpy()
#            t1=t1.to("cpu").numpy()
#            data=data.to("cpu").numpy()
#            
#            np.save('batch_1_pre'+str(batch_idx),p1)
#            np.save('batch_1_gt'+str(batch_idx),t1)
#            np.save('batch_1_imgs'+str(batch_idx),data)
            
            torchvision.utils.save_image(p1, f"{folder1}/{label[0]}_pre.png") 
            torchvision.utils.save_image(t1, f"{folder2}/{label[0]}_seg_gt.png")
            torchvision.utils.save_image(data, f"{folder3}/{label[0]}_img.png")
            

    print('total dice score is')
    print(dice_score1)
    print(f"Dice score for Segmentation of LA: {dice_score1/len(val_loader)}")
    #print(f"Dice score for Boundry: {dice_score2/Total_samples}")
    #print(f"Dice score for scars: {dice_score3/Total_samples}")
    #print('overall is  ',dice_score/3)
