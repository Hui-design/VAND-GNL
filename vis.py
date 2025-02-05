import numpy as np
import random
import torch
from torchvision import transforms
from PIL import ImageFilter
from sklearn.manifold import TSNE 
import seaborn as sns 
import matplotlib.pyplot as plt
import cv2, pdb, os
from PIL import Image
from torch.nn import functional as F
import pandas as pd

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

resize_lib = {0: torch.nn.AdaptiveAvgPool2d((64, 64)),
              1: torch.nn.AdaptiveAvgPool2d((32, 32)) ,
              2: torch.nn.AdaptiveAvgPool2d((16, 16))}  # 注意：可以尝试不同的下采样方式

def vis_anomaly_images(imgs, obj, mask_ori=None):
    image_mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    image_std = np.array(IMAGENET_STD).reshape(1,1,3)
    B, C, H, W = imgs.shape
    os.makedirs(f'imgs/{obj}', exist_ok=True)
    # masks_list = [resize_lib[i](mask_ori) for i in range(3)]
    for i in range(B):
        img = imgs[i].permute(1,2,0).cpu().numpy() # [H,W,3]
        img = ((img * image_std + image_mean)*255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f'imgs/{obj}/img{i}.jpg')

        # # pdb.set_trace()
        # for level in range(3):
        #     mask = (masks_list[level][i][0]>0.3).float().cpu().numpy() # [H,W]
        #     mean = mask.mean()
        #     mask = (mask * 255).astype(np.uint8)
        #     mask = Image.fromarray(mask)
        #     mask.save(f'imgs/{obj}/mask{i}_{level}.jpg')

        # mask_ = (mask_ori[i].permute(1,2,0).cpu().numpy().squeeze() * 255).astype(np.uint8)
        # mask_ = Image.fromarray(mask_)
        # mask_.save(f'imgs/{obj}/mask{i}.jpg')

def vis_gt(gt, name):
    gt = Image.fromarray((gt* 255).astype(np.uint8) , mode='L')
    gt.save(f'{name}.png')

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, _class_, alpha=0.5):
    # pdb.set_trace()
    image_mean = np.array(IMAGENET_MEAN).reshape(1,1,3)
    image_std = np.array(IMAGENET_STD).reshape(1,1,3)
    img = image[0].permute(1,2,0).cpu().numpy() # [H,W,3]
    img = ((img * image_std + image_mean)*255).astype(np.uint8)
    # img_size = scoremap.shape[0]
    
    # image = cv2.cvtColor(cv2.resize(cv2.imread(rgb_path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
    # mask = normalize(scoremap.cpu().numpy()[0,0])
    # vis = apply_ad_scoremap(vis, mask)
    np_image = np.asarray(img, dtype=float)
    # scoremap = scoremap.cpu().numpy()[0,0]
    scoremap[scoremap>1] = 1
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    scoremap = (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
    
    vis = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)  # BGR
    save_vis = f'imgs/map/{_class_}'
    if not os.path.exists(save_vis):
        os.makedirs(save_vis)
    cv2.imwrite(f'{save_vis}/map.png', vis)
    x = Image.fromarray(img)
    x.save(f'{save_vis}/img.png')
