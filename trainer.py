import os
import datetime
import shutil

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np

from model import CNets
from utils import get_transformed_imgs_2, get_mask


class TransformTrainer(object):
    def __init__(self, config, trainloader, valloader):
        self.config = config
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        self.best_val = 1.

        # Init networks
        self.model = CNets(2,32,2)
        self.model = self.model.to(self.device)
        
        # Defining loss criterions
        self.loss_fn = nn.MSELoss()
        self.loss_fn.to(self.device)

        # Defining optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=config['gamma'])

        #Init tensorboard writer
        log_path = os.path.join(config['log_root'], self.time_str)
        self.writer = SummaryWriter(log_path)


    """
    Save checkpoint
    """   
    def save_model(self, epoch, is_best=False):
        
        file_path = os.path.join(self.config['model_root'], self.time_str, f"Estimator_{epoch}.pth")
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)       
        torch.save(self.model.state_dict(), file_path)
        
        if is_best:
            best_path = os.path.join(self.config['model_root'], self.time_str, f"Estimator_best.pth")
            shutil.copyfile(file_path, best_path)



    """
    Validation function
    """
    def validate(self):
        with torch.no_grad():
            self.model.eval()

            total_loss = 0.
            for img, target in self.valloader:
                img_g = img.to(self.device)
                target_g = target[:,2:4].to(self.device)
                output = self.model(img_g)
                loss = self.loss_fn(output, target_g).clamp(max=10)

                total_loss += loss.item() * target_g.size(0)

        return total_loss / len(self.valloader.dataset)



    """
    Train function
    """
    def train(self):

        for epoch in range(self.config['num_epochs']+1):
            self.model.train()
            
            img, target = next(iter(self.trainloader))
                
            self.optimizer.zero_grad()

            img_g = img.to(self.device)
            target_g = target[:,2:4].to(self.device)
            output = self.model(img_g)
            loss = self.loss_fn(output, target_g)

            loss.backward()
            self.optimizer.step()


            if epoch % self.config['val_cadence'] == 0: 
                val_loss = self.validate()
                self.writer.add_scalars('Translation', {
                    'train': loss,
                    'val': val_loss}, epoch)
                print(f'Epoch: {epoch}/{self.config["num_epochs"]}, trn: {loss:.4f}, val: {val_loss:.4f}.')


            if epoch % self.config['save_cadence'] == 0:
                self.best_val = min(val_loss, self.best_val)
                self.save_model(epoch, val_loss==self.best_val)

            self.scheduler.step()

        self.writer.close()



class IdentifierTrainer(TransformTrainer):
    def __init__(self, config, trainloader, valloader):
        super().__init__(config=config, trainloader=trainloader,valloader=valloader)

        # Init Regressor networks
        self.regressor = CNets(2,32,1)
        dict_path = config['estimator_path']
        d = torch.load(dict_path, map_location=torch.device('cpu'))
        self.regressor.load_state_dict(d)
        self.regressor.to(self.device)

        self.mask = get_mask()
        self.mask = self.mask.to(self.device)
        
        # Defining loss criterions
        self.loss_fn = nn.BCELoss()
        self.loss_fn.to(self.device)
    

    """
    Save checkpoint
    """   
    def save_model(self, epoch, is_best=False):
        
        file_path = os.path.join(self.config['model_root'], self.time_str, f"Identifier_{epoch}.pth")
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)       
        torch.save(self.model.state_dict(), file_path)
        
        if is_best:
            best_path = os.path.join(self.config['model_root'], self.time_str, f"Identifier_best.pth")
            shutil.copyfile(file_path, best_path)



    """
    Train function
    """
    def train(self):

        for epoch in range(self.config['num_epochs']+1):
            self.model.train()
            self.optimizer.zero_grad()
            
            img, target = next(iter(self.trainloader))
            img_g = img.to(self.device)

            with torch.no_grad():
                self.regressor.eval()
                degrees_hat = self.regressor(img_g).to(torch.device('cpu'))
                img_hat = get_transformed_imgs_2(img.size(0), degrees_hat*90, img[:,0:1,:,:])
                img_hat_g = img_hat.to(self.device)
                imgs_out = torch.cat((img_g[:,1:,:,:], img_hat_g), dim=1)
                imgs_out = imgs_out * self.mask

            target_g = target[:,0:1].to(self.device)
            output = self.model(imgs_out)
            loss = self.loss_fn(torch.sigmoid(output), target_g)

            loss.backward()
            self.optimizer.step()


            if epoch % self.config['val_cadence'] == 0: 
                #val_loss = self.validate()
                self.writer.add_scalar('Identification', loss, epoch)
                print(f'Epoch: {epoch}/{self.config["num_epochs"]}, trn: {loss:.4f}.')


            if epoch % self.config['save_cadence'] == 0:
                self.best_val = min(loss, self.best_val)
                self.save_model(epoch, loss==self.best_val)


            self.scheduler.step()

        self.writer.close()