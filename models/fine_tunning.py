def main():
    model = UNet().to(device=DEVICE,dtype=torch.float)   
    
    
    ## Fine Tunnning Part ###
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=0)
    weights_paths="/data/home/acw676/seg_//UNET_n.pth.tar"
    checkpoint = torch.load(weights_paths,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
   
   ########################
   
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    
    loss_fn1 =IoULoss()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,scaler)
        
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
