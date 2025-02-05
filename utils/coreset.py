import torch
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import random_projection
from sklearn import linear_model


def get_mini_memory(patch_lib, num_mini=100):  # modify
    coreset_idx = get_coreset_idx_randomp(patch_lib, n=num_mini, eps=0.9)
    patch_lib = patch_lib[coreset_idx]   # 计算距离的特征需要归一化
    return patch_lib 


def get_coreset_idx_randomp(z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
    # print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    # try:
    #     transformer = random_projection.SparseRandomProjection(eps=eps, random_state=0)
    #     z_lib = torch.tensor(transformer.fit_transform(z_lib))

    #     print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
    # except ValueError:
    #     print("   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx + 1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in range(n - 1):
        distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
        min_distances = torch.minimum(distances, min_distances)  # iterative step
        select_idx = torch.argmax(min_distances)  # selection step

        # bookkeeping
        last_item = z_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))
    return torch.stack(coreset_idx)
