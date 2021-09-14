import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import optim
import torch.nn as nn

from hednet import HedNet
from utils.dataset import BasicDataset

from tqdm import tqdm

def train_net(net):
    """ funcion de entrenamiento """
    # hiperparametros:
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.96)

    # loss history:
    train_losses = []
    val_losses = []

    # early stopping vars:
    best_prec1 = 1e6
    epochs_no_improve = 0
    n_epochs_stop = 10

    # train loop:
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Train Epoch {epoch+1}/{epochs}') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch['image'], batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item() * imgs.size(0)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                
        train_losses.append(epoch_loss/len(train))
        
        #validacion:
        net.eval()
        epoch_loss = 0
        
        with tqdm(total=n_val, desc=f'Val Epoch {epoch+1}/{epochs}') as pbar:
            for batch in val_loader:
                imgs, true_masks = batch['image'], batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    mask_pred = net(imgs)
                    loss = criterion(mask_pred, true_masks)
                
                epoch_loss += loss.item() * imgs.size(0)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                
        scheduler.step()
        val_losses.append(epoch_loss/len(val))

        # se guarda el modelo si es mejor que el anterior:
        prec1 = epoch_loss/n_val
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        
        if is_best:
            torch.save(net.state_dict(), 'MODEL.pth') 
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                break    

    print(f'The best MSE: {min(val_losses)}')    



if __name__ == "__main__":

    # definiendo los hiperparametros:
    epochs = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_percent = 0.2
    img_scale = 1
    dir_img = 'data_dm_overlapping/imgs/'
    dir_mask = 'data_dm_overlapping/masks/'
    batch_size = 4
    lr = 0.0001

    # definiendo el dataset:
    dataset = BasicDataset(dir_img, dir_mask, img_scale, transforms=transforms.Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
                                                                    ]), mask_h5=True)

    # definimos los conjunto de entrenamiento, validacion y pruebas:
    n_val = int(len(dataset) * val_percent)
    n_test = n_val
    n_train = len(dataset) - n_val - n_test
    train, val, test = random_split(dataset, [n_train, n_val, n_test])

    # batch loader:
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)


    net = HedNet(n_channels=3, n_classes=1, bilinear=False)
    net.to(device=device)

    train_net(net)