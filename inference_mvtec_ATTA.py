import torch
from dataset import get_data_transforms
# from resnet_TTA import  wide_resnet50_2_TTA
from resnet import wide_resnet50_2
from de_resnet import  de_wide_resnet50_2
from dataset import MVTecDataset, MVTecDatasetOOD
from test import  evaluation_ATTA
from tqdm import tqdm
import numpy as np
from model_utils import SegmentationNet
import argparse
import pdb, os


def test_mvtec(_class_, epoch, root):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Class: ', _class_)
    data_transform, gt_transform = get_data_transforms(256, 256)

    #load data
    test_path_id = os.path.join(root, _class_) #update here
    test_path_brightness = os.path.join(root.replace('mvtec', 'mvtec_brightness'), _class_) 
    test_path_constrast = os.path.join(root.replace('mvtec', 'mvtec_contrast'), _class_) 
    test_path_defocus_blur = os.path.join(root.replace('mvtec', 'mvtec_defocus_blur'), _class_) 
    test_path_gaussian_noise = os.path.join(root.replace('mvtec', 'mvtec_gaussian_noise'), _class_)  
    test_path_rot = os.path.join(root.replace('mvtec', 'mvtec_rot'), _class_) 
    ckp_path = './checkpoints/' + 'mvtec_KD_' + str(_class_) + f'_best.pth'
    test_data_id = MVTecDataset(root=test_path_id, transform=data_transform, gt_transform=gt_transform,
                             phase="test")
    test_data_brightness = MVTecDatasetOOD(root=test_path_brightness, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_constrast = MVTecDatasetOOD(root=test_path_constrast, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_defocus_blur = MVTecDatasetOOD(root=test_path_defocus_blur, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_gaussian_noise = MVTecDatasetOOD(root=test_path_gaussian_noise, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)
    test_data_rot = MVTecDatasetOOD(root=test_path_rot, transform=data_transform, gt_transform=gt_transform,
                             phase="test", _class_=_class_)

    test_dataloader_id = torch.utils.data.DataLoader(test_data_id, batch_size=1, shuffle=False)
    test_dataloader_brightness = torch.utils.data.DataLoader(test_data_brightness, batch_size=1, shuffle=False)
    test_dataloader_constrast = torch.utils.data.DataLoader(test_data_constrast, batch_size=1, shuffle=False)
    test_dataloader_defocus_blur = torch.utils.data.DataLoader(test_data_defocus_blur, batch_size=1, shuffle=False)
    test_dataloader_gaussian_noise = torch.utils.data.DataLoader(test_data_gaussian_noise, batch_size=1, shuffle=False)
    test_dataloader_rot = torch.utils.data.DataLoader(test_data_rot, batch_size=1, shuffle=False)

    #load model
    # encoder, bn = wide_resnet50_2_TTA(pretrained=True)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    model = SegmentationNet(inplanes=1792).to(device)

    #load checkpoint
    ckp = torch.load(ckp_path)
    decoder.load_state_dict(ckp['decoder'], strict=False)
    bn.load_state_dict(ckp['bn'])
    seg_ckpt = torch.load('checkpoints/'  + 'mvtec_DINL_' + str(_class_) + f'_best.pth')
    model.load_state_dict(seg_ckpt['seg'])

    lamda = 0.5

    # print('not use seg_model now!!!')
    # model = None

    sp_results, px_results = [], []
    F1_sp_results, F1_px_results = [], []
    auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_id, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_, seg_model=model)
    sp_results.append(round(auroc_sp, 4))
    px_results.append(round(auroc_px, 4))
    F1_sp_results.append(round(F1_sp, 4))
    F1_px_results.append(round(F1_px, 4))

    auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_brightness, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_, seg_model=model)
    sp_results.append(round(auroc_sp, 4))
    px_results.append(round(auroc_px, 4))
    F1_sp_results.append(round(F1_sp, 4))
    F1_px_results.append(round(F1_px, 4))

    auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_constrast, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_, seg_model=model)
    sp_results.append(round(auroc_sp, 4))
    px_results.append(round(auroc_px, 4))
    F1_sp_results.append(round(F1_sp, 4))
    F1_px_results.append(round(F1_px, 4))

    auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_defocus_blur, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_, seg_model=model)
    sp_results.append(round(auroc_sp, 4))
    px_results.append(round(auroc_px, 4))
    F1_sp_results.append(round(F1_sp, 4))
    F1_px_results.append(round(F1_px, 4))

    auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_gaussian_noise, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_, seg_model=model)
    sp_results.append(round(auroc_sp, 4))
    px_results.append(round(auroc_px, 4))
    F1_sp_results.append(round(F1_sp, 4))
    F1_px_results.append(round(F1_px, 4))

    auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_rot, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='mvtec', _class_=_class_, seg_model=model)
    sp_results.append(round(auroc_sp, 4))
    px_results.append(round(auroc_px, 4))
    F1_sp_results.append(round(F1_sp, 4))
    F1_px_results.append(round(F1_px, 4))

    print(sp_results, px_results, F1_sp_results, F1_px_results)

    return np.array(sp_results), np.array(px_results),  np.array(F1_sp_results), np.array(F1_px_results)

item_list = [
            'carpet', 'leather', 'grid', 
            'tile', 
            'wood', 
            'bottle', 'hazelnut', 'cable',
            'capsule', 'pill', 
            'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper'
            ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/path/to/mvtec')
    args = parser.parse_args()
    # for epoch in [19,39,59,79,99]:
    for epoch in [79]:
        print(epoch)
        sp_result = []
        px_result = []
        F1_sp_result = []
        F1_px_result = []
        for i in tqdm(item_list):
            sp, px, F1_sp, F1_px = test_mvtec(i, epoch, args.data_root)
            print('about two minutes')
            sp_result.append(sp)
            px_result.append(px)
            F1_sp_result.append(F1_sp)
            F1_px_result.append(F1_px)
            print('===============================================')
        # pdb.set_trace()
        print(np.array(sp_result).mean(axis=0))
        print(np.array(px_result).mean(axis=0))
        print(np.array(F1_sp_result).mean(axis=0))
        print(np.array(F1_px_result).mean(axis=0))
