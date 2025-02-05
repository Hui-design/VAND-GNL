from PIL import Image
import os
import glob
from PIL import Image
from imagecorruptions import corrupt
import numpy as np
import pdb
from tqdm import tqdm
from dataset import rotation
import argparse

item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
             'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']

def generate(root):
    # pdb.set_trace()
    for type_cor in ['brightness','contrast','defocus_blur','gaussian_noise']:
        for _class_ in item_list:
            path_orginal = root + _class_ + '/' + 'test' #path to the test set of original mvtec 
            type_sets = glob.glob(path_orginal+'/*/')
            for type in type_sets:
                path_type = type
                path_type_new = path_type.replace('mvtec', 'mvtec_'+type_cor)
                print(path_type_new)
                isExist = os.path.exists(path_type_new)
                if not isExist:
                    os.makedirs(path_type_new)
                    print("The new directory is created!")
                image_names = glob.glob(path_type + '/*.png')
                for image_name in image_names:
                    path_to_image = image_name
                    print(path_to_image)
                    image = Image.open(path_to_image)
                    image = np.array(image)
                    corrupted = corrupt(image, corruption_name=type_cor, severity=3)
                    im = Image.fromarray(corrupted)
                    im.save(path_to_image.replace('mvtec', 'mvtec_'+type_cor))


    type_cor = 'rot'
    for _class_ in item_list:
        path_orginal = root + _class_ + '/' + 'test' #path to the test set of original mvtec 
        gt_orginal = root + _class_ + '/' + 'ground_truth' #path to the test set of original mvtec 
        type_sets = glob.glob(path_orginal+'/*/')
        for type in type_sets:
            path_type = type
            path_type_new = path_type.replace('mvtec', 'mvtec_'+type_cor)
            print(path_type_new)
            gt_path_new = path_type_new.replace('test', 'ground_truth')
            gt_path = path_type.replace('test', 'ground_truth')
            os.makedirs(path_type_new, exist_ok=True)
            os.makedirs(gt_path_new, exist_ok=True)
            # pdb.set_trace()
            image_names = sorted(glob.glob(path_type + '/*.png'))
            gt_names = sorted(glob.glob(gt_path + '/*.png'))
            for i, image_name in enumerate(image_names):
                path_to_image = image_name
                print(path_to_image)
                image = Image.open(path_to_image).convert('RGB')
                if len(gt_names) != 0:
                    path_to_gt = gt_names[i]
                    gt = Image.open(path_to_gt).convert('L')    
                else:
                    # pdb.set_trace()
                    path_to_gt = path_to_image.replace('test', 'ground_truth')
                    # path_to_gt[-4:] = '_mask.png'
                    gt = Image.fromarray(np.zeros_like(np.array(image)[:,:,0]).astype(np.uint8), mode='L')
                im, gt = rotation(image, gt)
                im.save(path_to_image.replace('mvtec', 'mvtec_'+type_cor))
                gt.save(path_to_gt.replace('mvtec', 'mvtec_'+type_cor))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/path/to/mvtec')
    args = parser.parse_args()
    generate(args.data_path)