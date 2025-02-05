import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
from matplotlib import image
import torchvision.transforms as T
from matplotlib import pyplot as plt
from os import listdir
from torchvision import transforms
from sklearn.metrics import precision_recall_curve
from unet import Unet
from vis import *
import torch
from model_utils import l2_normalize
import pdb

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def evaluation_ATTA(encoder, bn, decoder, dataloader,device, type_of_test, img_size, lamda=0.5, dataset_name='mnist', _class_=None, seg_model=None):
    bn.eval()
    decoder.eval()
    if seg_model is not None:
        seg_model.eval()
    gt_list_sp, gt_list_px = [], []
    pr_list_sp, pr_list_px = [], []

    if dataset_name == 'mvtec':
        link_to_normal_sample = '/data4/tch/AD_data/mvtec/' + _class_ + '/train/good/000.png' #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name != 'mnist':
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)
        ])
    else:
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    normal_image = trans(normal_image)
    normal_image = torch.unsqueeze(normal_image, 0)

    # model = Unet().to(device)
    # model.load_state_dict(torch.load(f'checkpoints/recons{_class_}_99.pth')['unet'])

    with torch.no_grad():
        for sample in dataloader:
            img, gt = sample[0], sample[1]
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            label = int(torch.sum(gt) != 0)

            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            normal_image = normal_image.to(device)
            img = img.to(device)
            # img = my_clamp(model(img))
            # inputs = encoder(img, normal_image, type_of_test, lamda=lamda)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map_ori = gaussian_filter(anomaly_map, sigma=4)
            # sample_score = anomaly_map_ori.max()
            if seg_model is not None:
                outputs_teacher_aug = [l2_normalize(output_t) for output_t in inputs]
                outputs_student_aug = [l2_normalize(output_s) for output_s in outputs]
                output = torch.cat(
                [
                    F.interpolate(
                        -output_t * output_s,
                        size=outputs_student_aug[0].size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
                ], dim=1)
                output_segmentation = seg_model(output)
                anomaly_map = F.interpolate(
                    output_segmentation,
                    size=gt.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().cpu().numpy()
                # pdb.set_trace()
                # vis_gt(np.clip(anomaly_map, a_min=0, a_max=1), 'pred')
                # vis_gt(gt[0,0].cpu().numpy(), 'gt')
                # vis_anomaly_images(img, 'fe')
                anomaly_map = anomaly_map + anomaly_map_ori
            
            # if int(label) !=0:
                # apply_ad_scoremap(img, gt.cpu().numpy()[0,0], _class_)
                # apply_ad_scoremap(img, anomaly_map, _class_)
                # pdb.set_trace()
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(int(label))
            pr_list_sp.append(np.max(anomaly_map))

        precision, recall, thresholds = precision_recall_curve(gt_list_sp, pr_list_sp) # sample
        F1_scores = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
        sp_F1 = np.max(F1_scores)
        precision, recall, thresholds = precision_recall_curve(gt_list_px, pr_list_px) # pixel
        F1_scores = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
        px_F1 = np.max(F1_scores)

        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)

    return auroc_sp, auroc_px, sp_F1, px_F1



