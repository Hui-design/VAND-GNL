import torch
from torchvision.datasets import ImageFolder
from resnet import wide_resnet50_2
# from resnet_TTA import  wide_resnet50_2_TTA
from de_resnet import de_wide_resnet50_2
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataset import SynAugMixDatasetMVTec
from tqdm import tqdm
from unet import Unet
import os
import pdb
from utils.utils import setup_seed
from vis import *
from loss import focal_loss, l1_loss
from model_utils import SegmentationNet, l2_normalize
from dataset import get_data_transforms, MVTecDataset, MVTecDatasetOOD
from test import  evaluation_ATTA
import argparse
setup_seed(111)


def train(_class_, root, dtd_root):
    print(_class_)
    epochs = 100
    learning_rate = 0.005  # 0.005
    batch_size = 32       # 16
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


    resize_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    data_transform, gt_transform = get_data_transforms(256, 256)
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
    ])


    train_path = os.path.join(root, _class_ + '/train') #update here
    test_path_id =  os.path.join(root, _class_ )#update here
    train_data = SynAugMixDatasetMVTec(train_path, preprocess, dtd_root)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_data_id = MVTecDataset(root=test_path_id, transform=data_transform, gt_transform=gt_transform,phase="test")
    test_dataloader_id = torch.utils.data.DataLoader(test_data_id, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp_path = './checkpoints/' + 'mvtec_KD_' + str(_class_) + f'_best.pth'

    # load model
    # encoder_TTA, _ = wide_resnet50_2_TTA(pretrained=True)
    # encoder_TTA = encoder_TTA.to(device)
    # encoder_TTA.eval()

    # load checkpoint
    ckp = torch.load(ckp_path)
    decoder.load_state_dict(ckp['decoder'], strict=False)
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    model = SegmentationNet(inplanes=1792).to(device)
    optimizer = torch.optim.SGD(
        [
            {"params": model.res.parameters(), "lr": 0.1},
            {"params": model.head.parameters(), "lr": 0.01},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    best_F1 = -1
    for epoch in tqdm(range(epochs)):
        model.train()
        loss_list = []
        loss_struct_list = []
        for normal, augmix_img, mask in train_dataloader:
            # pdb.set_trace()
            with torch.no_grad():
                augmix_img = augmix_img.to(device)
                mask = mask.to(device)
                inputs_augmix = encoder(augmix_img)
                bn_augmix = bn(inputs_augmix)
                outputs_augmix = decoder(bn_augmix)
                outputs_teacher_aug = [l2_normalize(output_t) for output_t in inputs_augmix]
                outputs_student_aug = [l2_normalize(output_s) for output_s in outputs_augmix]
            
            # pdb.set_trace()
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

            output_segmentation = model(output)

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            focal_loss_val = focal_loss(output_segmentation, mask, gamma=4)
            l1_loss_val = l1_loss(output_segmentation, mask)
            loss = focal_loss_val + l1_loss_val
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(np.mean(loss_list))
        if (epoch + 1) % 10 == 0 :
            model.eval()
            auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_id, device,
                                               type_of_test='EFDM_test', img_size=256, lamda=0.5, dataset_name='mvtec',
                                               _class_=_class_, seg_model=model)
            print(auroc_sp, auroc_px, F1_sp, F1_px)
            if (F1_px+F1_sp)/2 > best_F1:
                best_F1 = (F1_px+F1_sp)/2
                print('save')
                os.makedirs('./checkpoints', exist_ok=True)
                ckp_path = './checkpoints/' + 'mvtec_DINL_' + str(_class_) + '_best.pth'
                torch.save({'seg': model.state_dict()}, ckp_path)
        
    return


if __name__ == '__main__':
    item_list = [
                'carpet', 'leather', 'grid', 
                'wood', 'bottle', 'hazelnut', 'cable',
                'capsule', 'pill', 'transistor',
                'screw','toothbrush', 'zipper', 'tile'
                ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/path/to/mvtec')
    parser.add_argument('--aux_path', type=str, default='/path/to/dtd/images')
    args = parser.parse_args()
    for i in item_list:
        train(i, args.data_root, args.aux_path)

