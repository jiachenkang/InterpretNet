import math
import random

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torchvision.transforms.functional import _get_inverse_affine_matrix
from torchvision.datasets import MNIST

from utils import get_mask

class NoiseBW(IterableDataset):

    def __init__(self, original_size=(28,28), img_size=(28,28), rotation=0., translation=0., scale=0., batch_size=256, identifier=False):
        super().__init__()

        self.original_size = original_size
        self.img_size = img_size

        self.rotation = rotation
        self.translation = translation
        self.scale = scale

        self.batch_size = batch_size

        self.identifier = identifier
        self.mask = get_mask(img_size)

    def __iter__(self):

        while True:
            targets_tranform = (torch.rand(self.batch_size, 4) - 0.5) * 2 #0:rotation,1:translate_x,2:translate_y,3:scale
            targets_id = torch.randint(2, (self.batch_size,))
            
            degrees = targets_tranform[:,0] * self.rotation
            translates_x = targets_tranform[:,1] * self.translation / self.original_size[0] * 2 # ratio to half width
            translates_y = targets_tranform[:,2] * self.translation / self.original_size[1] * 2 # ratio to half height
            scales = targets_tranform[:,3] * self.scale + 1.0

            imgs = np.random.choice([0.0, 1.0], size=(self.batch_size, 1, *self.original_size), p=(0.7, 0.3))
            imgs_t = torch.from_numpy(imgs).to(torch.float32)

            matrix = torch.zeros(self.batch_size, 2, 3)
            for i in range(self.batch_size):
                matrix_inv = _get_inverse_affine_matrix((0,0),degrees[i],(translates_x[i],translates_y[i]),scales[i],(0.0,0.0))
                matrix[i] = torch.tensor([
                                    matrix_inv[:3],
                                    matrix_inv[3:6]
                                ], dtype=torch.float32)
            affine = F.affine_grid(matrix, imgs_t.size(), align_corners=False)
            imgs_t_transform = F.grid_sample(imgs_t, affine, mode='bilinear', align_corners=False)

            pad_x = int((self.original_size[0] - self.img_size[0])/2)
            pad_y = int((self.original_size[1] - self.img_size[1])/2)
            imgs_t_out = imgs_t[:,:,pad_x:pad_x+self.img_size[0],pad_y:pad_y+self.img_size[1]]#crop in center
            imgs_t_transform_out = imgs_t_transform[:,:,pad_x:pad_x+self.img_size[0],pad_y:pad_y+self.img_size[1]]#crop in center

            if self.identifier:
                ndx_neg = torch.nonzero(targets_id==0).squeeze(1) #0:different image(negative), 1: same image(positive)
                imgs_neg = np.random.choice([0.0, 1.0], size=(len(ndx_neg), 1, *self.img_size), p=(0.7, 0.3))
                imgs_neg_t = torch.from_numpy(imgs_neg).to(torch.float32)
                imgs_t_out[ndx_neg] = imgs_neg_t

            
            imgs_out = torch.cat((imgs_t_out, imgs_t_transform_out), dim=1)
            imgs_out = imgs_out * self.mask * 2 - 1

            targets_out = torch.cat((targets_id.unsqueeze(1), targets_tranform), dim=1)

            yield imgs_out, targets_out




class TMNIST(MNIST):
    def __init__(self, root, rotation=0., translation=0., scale=0., resample=Image.BILINEAR, fillcolor=0, train=True, transform=None, target_transform=None, download=False, identifier=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self.shear = (0.0, 0.0)

        self.resample = resample
        self.fillcolor = fillcolor

        self.targets_tranform = (torch.rand(len(self.data), 4) - 0.5) * 2 #0:rotation,1:translate_x,2:translate_y,3:scale
        self.identifier = identifier

    def __getitem__(self, index):
        img = self.data[index]
        target_tranform = self.targets_tranform[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img.numpy(), mode='L')

        degree =  target_tranform[0] * self.rotation
        translate_x = target_tranform[1] * self.translation
        translate_y = target_tranform[2] * self.translation
        scale = target_tranform[3] * self.scale + 1.0

        # affine transformation on image
        output_size = img.size
        center = (img.size[0] * 0.5, img.size[1] * 0.5)
        matrix = _get_inverse_affine_matrix(center, degree, (translate_x,translate_y), scale, self.shear)
        img_transform = img.transform(output_size, Image.AFFINE, matrix, self.resample, fillcolor=self.fillcolor)


        if self.identifier:
            t_id = torch.randint(2, (1,))
            if not t_id: #0:different, 1:same
                ndx = int(torch.randint(len(self.data), (1,)))
                img = self.data[ndx]
                img = Image.fromarray(img.numpy(), mode='L')
        else:
            t_id = torch.tensor([self.targets[index],])


        if self.transform is not None:
            img = self.transform(img)
            img_transform = self.transform(img_transform)
        
        img_cat = torch.cat((img,img_transform), dim=0)

        t_cat = torch.cat((t_id, target_tranform), dim=0)

        return img_cat, t_cat