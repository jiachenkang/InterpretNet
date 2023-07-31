import random
import math
import collections

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import _get_inverse_affine_matrix
import torch.nn.functional as F


def get_params(degree, translate, scale_ranges, shears):
    """Get parameters for affine transformation
    Returns:
        sequence: params to be passed to the affine transformation
    """
    angle = random.uniform(-degree, degree)

    if translate is not None:
        translations = (random.uniform(-translate, translate),
                        random.uniform(-translate, translate))
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    else:
        scale = 1.0

    if shears is not None:
        shear = (random.uniform(shears[0], shears[1]), 0.0)
    else:
        shear = (0.0, 0.0)

    return angle, translations, scale, shear

def get_transformed_imgs(img, degrees, translate, scale, shear, resample, fillcolor):

    # affine transformation on image1
    ret1 = get_params(degrees, (translate[0]*0.6, translate[1]*0.6), scale, (shear[0]*0.5, shear[1]*0.5), img.size)
    output_size = img.size
    center1 = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix1 = _get_inverse_affine_matrix(center1, *ret1)
    img1 = img.transform(output_size, Image.AFFINE, matrix1, resample, fillcolor=fillcolor)

    # affine transformation on image2
    ret2 = get_params(degrees, translate, scale, shear, img.size)
    center2 = (img.size[0] * 0.5 + 0.5 + ret1[1][0], img.size[1] * 0.5 + 0.5 + ret1[1][1])
    matrix2 = _get_inverse_affine_matrix(center2, *ret2)
    img2 = img1.transform(output_size, Image.AFFINE, matrix2, resample, fillcolor=fillcolor)

    aff_para = [math.cos(math.radians(ret2[0])),
                math.sin(math.radians(ret2[0])),
                ret2[1][0]/translate[0]/output_size[0],
                ret2[1][1]/translate[1]/output_size[1],
                ret2[2]*2./(scale[1]-scale[0])-(scale[0]+scale[1])/(scale[1]-scale[0]),
                ret2[3][0]*2./(shear[1]-shear[0])-(shear[0]+shear[1])/(shear[1]-shear[0])]

    return img1, img2, aff_para

def get_ex_aff_para(aff_para, translate, scale, shear, img_size):
    degree_cos = math.degrees(math.acos(aff_para[0]))
    degree_sin = math.degrees(math.asin(aff_para[1]))
    if degree_sin < 0:
        degree_cos *= -1
    
    ex_aff_para = [(degree_cos + degree_sin) / 2,
                (round(aff_para[2]*img_size[0]*translate[0]), round(aff_para[3]*img_size[1]*translate[1])),
                (aff_para[4]*(scale[1]-scale[0])+(scale[0]+scale[1]))/2.,
                ((aff_para[5]*(shear[1]-shear[0])+(shear[0]+shear[1]))/2., 0.0)]
    
    return ex_aff_para

def get_paired_imgs(img_p, img_n, degrees, translate, scale, shear, resample, fillcolor):
    img_p1, img_p2, aff_para = get_transformed_imgs(img_p, degrees, translate, scale, shear, resample, fillcolor)
    img_n1, _, _ = get_transformed_imgs(img_n, degrees, translate, scale, shear, resample, fillcolor)
    if random.random() > 0.5:
        target = [1.0,] + aff_para
        return img_p1, img_p2, target
    else:
        target = [0.0,] * 7
        return img_n1, img_p2, target

def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=np.int)
    return labels_to_indices

def get_transformed_imgs_2(batch_size, degrees, imgs_t):
    matrix = torch.zeros(batch_size, 2, 3)

    for i, degree in enumerate(degrees):
        c = math.cos(math.radians(degree.item()))
        s = math.sin(math.radians(degree.item()))
        matrix[i,:,:2] = torch.Tensor([[c, -s],[s, c]])

    affine_t = F.affine_grid(matrix, imgs_t.size(), align_corners=False)
    imgs_1 = F.grid_sample(imgs_t, affine_t, mode='bilinear', align_corners=False)

    return imgs_1
    

def get_mask(img_size):
    x = np.arange(0, img_size[0], 1, np.float32)
    y = np.arange(0, img_size[1], 1, np.float32)
    y = y[:, np.newaxis] 
    mask = (((x - img_size[0]/2 + 0.5) ** 2 + (y - img_size[1]/2 + 0.5) ** 2) < (img_size[0]/2) ** 2).astype(np.uint8)
    return torch.from_numpy(mask)