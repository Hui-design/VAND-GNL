import torch
from torchvision.datasets import ImageFolder
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataset import AugMixDatasetMVTec, MVTecDataset, get_data_transforms
from tqdm import tqdm
from unet import Unet
import os
import pdb
from utils.utils import setup_seed
from test import  evaluation_ATTA
from vis import *
import argparse

setup_seed(111)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        B, C, H, W = a[item].shape
        f_t = a[item].reshape(B,C,H*W)
        f_s = b[item].reshape(B,C,H*W)
        loss_i = 1 - cos_loss(f_t, f_s)  # [B,C,H*W] -> [B,H*W], 相似度越高，loss越小
        loss += torch.mean(loss_i.topk(k=10, dim=1)[0]) # 选择损失大的
        loss += torch.mean(1-cos_loss(a[item].reshape(a[item].shape[0],-1),
                                      b[item].reshape(b[item].shape[0],-1)))
    return loss

def loss_structure(a, b, smooth=1e-8):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        B, C, H, W = a[item].shape
        # pdb.set_trace()
        f_t = torch.exp(a[item]) / torch.exp(a[item]).sum(dim=(-1,-2), keepdim=True)  + smooth
        f_s = torch.exp(b[item]) / torch.exp(b[item]).sum(dim=(-1,-2), keepdim=True)  + smooth
        loss_i = f_t * torch.log(f_t / f_s + smooth) 
        loss += torch.mean(loss_i)
    return loss

def loss_function_last(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # for item in range(len(a)):
    #     # print(a[item].shape)
    #     # print(b[item].shape)
    #     # loss += 0.1*mse_loss(a[item], b[item])
    item = 0
    loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                    b[item].view(b[item].shape[0], -1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        # loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss

def my_clamp(x):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    max_list = (1-IMAGENET_MEAN)/IMAGENET_STD
    min_list = (-IMAGENET_MEAN)/IMAGENET_STD
    for i in range(3):
        # pdb.set_trace()
        x[:,i] = torch.clamp(x[:,i], min_list[i], max_list[i])
    return x

def train(_class_, root):
    print(_class_)
    epochs = 80
    learning_rate = 0.005  # 0.005
    batch_size = 16       # 16
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


    resize_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
    ])

    data_transform, gt_transform = get_data_transforms(256, 256)
    train_path = os.path.join(root, _class_ + '/train') #update here
    test_path_id =  os.path.join(root, _class_ )#update here
    train_data = ImageFolder(root=train_path, transform=resize_transform)
    train_data = AugMixDatasetMVTec(train_data, preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_data_id = MVTecDataset(root=test_path_id, transform=data_transform, gt_transform=gt_transform,phase="test")
    test_dataloader_id = torch.utils.data.DataLoader(test_data_id, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    # model = Unet().to(device)
    # model.load_state_dict(torch.load(f'checkpoints/recons{_class_}_99.pth')['unet'])

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))
    best_F1 = -1
    for epoch in tqdm(range(epochs)):
        bn.train()
        decoder.train()
        loss_list = []
        loss_struct_list = []
        for normal, augmix_img, gray_img in train_dataloader:
            # pdb.set_trace()
            with torch.no_grad():
                normal = normal.to(device)
                inputs_normal = encoder(normal)
                augmix_img = augmix_img.to(device)
                inputs_augmix = encoder(augmix_img)
                # gray_img = gray_img.to(device)
                # inputs_gray = encoder(gray_img)
            # pdb.set_trace()

            bn_normal = bn(inputs_normal)
            outputs_normal = decoder(bn_normal)  
            bn_augmix = bn(inputs_augmix)
            outputs_augmix = decoder(bn_augmix)
            # bn_gray = bn(inputs_gray)
            # outputs_gray = decoder(bn_gray)
            # pdb.set_trace()
            loss_normal = loss_function(inputs_normal, outputs_normal) + loss_function(inputs_augmix, outputs_augmix)
            loss_struct = loss_structure(inputs_normal, outputs_normal) + loss_structure(inputs_augmix, outputs_augmix)
            loss = loss_normal + loss_struct 
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_struct_list.append(loss_struct.item())
            loss_list.append(loss.item())
        print(np.mean(loss_list), np.mean(loss_struct_list))
        if (epoch + 1) % 10 == 0 :
            bn.eval()
            decoder.eval()
            auroc_sp, auroc_px, F1_sp, F1_px = evaluation_ATTA(encoder, bn, decoder, test_dataloader_id, device,
                                               type_of_test='EFDM_test', img_size=256, lamda=0.5, dataset_name='mvtec',
                                               _class_=_class_)
            print(auroc_sp, auroc_px, F1_sp, F1_px)
            if (F1_px+F1_sp)/2 > best_F1:
                best_F1 = (F1_px+F1_sp)/2
                print('save')
                os.makedirs('./checkpoints', exist_ok=True)
                ckp_path = './checkpoints/' + 'mvtec_KD_' + str(_class_) + '_best'  + '.pth'
                torch.save({'bn': bn.state_dict(),
                            'decoder': decoder.state_dict()}, ckp_path)
        
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
    args = parser.parse_args()
    for i in item_list:
        train(i, args.data_root)
