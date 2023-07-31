from __future__ import print_function
from __future__ import absolute_import
import os
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

from config import ex
from dataset import TMNIST, NoiseBW
from trainer import TransformTrainer, IdentifierTrainer

@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    print(_config)

    
    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and _config['gpu'] == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if _config['gpu'] >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_config['gpu'])

    # Creating data loaders
    mean = np.array([0.5, ])
    std = np.array([0.5, ])

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean,std)])

    train_set = NoiseBW(translation=_config['translation'], original_size=(40,40), identifier=_config['train_identifier']) #original_size=(40,40) for translation and scale, (64,64) for combined
    val_set = TMNIST('data/', translation=_config['translation'], train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=None, num_workers=_config['num_workers'])
    valloader = torch.utils.data.DataLoader(val_set, batch_size=_config['batch_size'], shuffle=False, num_workers=_config['num_workers'])
    
    if not _config['train_identifier']:
        # Training estimator
        trainer = TransformTrainer(_config, trainloader, valloader)
        trainer.train()
    else:
        # Training identifier
        trainer = IdentifierTrainer(_config, trainloader, valloader)
        trainer.train()
