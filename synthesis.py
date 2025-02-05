from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torch.utils.data import Dataset
import numpy as np
import pdb
from torch.utils.data import Dataset
from torchvision import transforms as T
import imgaug.augmenters as iaa
import albumentations as A
from utils.perlin import rand_perlin_2d_np
from utils.nsa import patch_ex
from utils.mvtec3d_util import *
import random
from utils.vis import vis_anomaly_images
import cv2

size = 256
augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
            ]
rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
    

def transform_image(image, fore_mask, anomaly_source_paths):
    anomaly_source_idx = torch.randint(0, len(anomaly_source_paths), (1,)).item()
    anomaly_source_path = anomaly_source_paths[anomaly_source_idx]
    # image = image.permute(1,2,0).numpy()
    if fore_mask is not None:
        fore_mask = fore_mask.permute(1,2,0).numpy()
    # normalize the image to 0.0~1.0
    augmented_image, anomaly_mask, has_anomaly = augment_image(image, anomaly_source_path, fore_mask)
    # augmented_image = np.transpose(augmented_image, (2, 0, 1))
    # image = np.transpose(image, (2, 0, 1))
    anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
    return augmented_image, anomaly_mask, has_anomaly

def augment_image(image, anomaly_source_path, fore_mask):
    aug = randAugmenter()
    perlin_scale = 6
    min_perlin_scale = 0

    anomaly_source_img = cv2.imread(anomaly_source_path)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(size, size))

    anomaly_img_augmented = aug(image=anomaly_source_img)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    
    if fore_mask is not None:
        while True:
            perlin_noise = rand_perlin_2d_np((size, size), (perlin_scalex, perlin_scaley))
            perlin_noise = rot(image=perlin_noise)
            threshold = 0.5
            # modify
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)
            perlin_thr = perlin_thr * fore_mask
            # pdb.set_trace()
            if perlin_thr.sum() > 4:
                break
    else:
        perlin_noise = rand_perlin_2d_np((size, size), (perlin_scalex, perlin_scaley))
        perlin_noise = rot(image=perlin_noise)
        threshold = 0.5
        # modify
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
    #
    # modify, '/255' 改成 imagenet 的归一化
    # 测试一下img_thr的最大值和image的最小值
    # img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
    # img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
    # image_mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    # image_std = np.array(IMAGENET_STD).reshape(1,1,3)
    # img_thr = (img_thr - image_mean) / image_std
    img_thr = anomaly_img_augmented
    # pdb.set_trace()

    beta = torch.rand(1).numpy()[0] * 0.8
    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)
    
    augmented_image = augmented_image.astype(np.float32)
    msk = (perlin_thr).astype(np.float32)
    augmented_image = msk * augmented_image + (1-msk)*image  # This line is unnecessary and can be deleted
    has_anomaly = 1.0

    no_anomaly = torch.rand(1).numpy()[0]
    if no_anomaly > 0.7:
        image = image.astype(np.float32)
        return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
    else:  # 0.7概率产生异常
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image  # This line is unnecessary and can be deleted
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

def randAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([augmenters[aug_ind[0]],
                        augmenters[aug_ind[1]],
                        augmenters[aug_ind[2]]]
                            )
    return aug