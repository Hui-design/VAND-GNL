from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from synthesis import transform_image
from imagecorruptions import corrupt
import pdb

IMAGE_SIZE = 256
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_data_transforms_augmix(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.AugMix(severity=10, all_ops=False),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms



class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type


class MVTecDatasetOOD(torch.utils.data.Dataset):
	def __init__(self, root, transform, gt_transform, phase, _class_):
		if phase == 'train':
			self.img_path = os.path.join(root, 'train')
		else:
			self.img_path = os.path.join(root, 'test')
			if root.endswith('/'):
				ori_path = root.replace(root.split('/')[-3], 'mvtec')
			else:
				ori_path = root.replace(root.split('/')[-2], 'mvtec')
			self.gt_path = os.path.join(f'{ori_path}', 'ground_truth')
			if 'mvtec_rot' in root:
				self.gt_path = os.path.join(root, 'ground_truth')
		self.transform = transform
		self.gt_transform = gt_transform
		# load dataset
		self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

	def load_dataset(self):

		img_tot_paths = []
		gt_tot_paths = []
		tot_labels = []
		tot_types = []

		defect_types = os.listdir(self.img_path)

		for defect_type in defect_types:
			if defect_type == 'good':
				img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
				img_tot_paths.extend(img_paths)
				gt_tot_paths.extend([0] * len(img_paths))
				tot_labels.extend([0] * len(img_paths))
				tot_types.extend(['good'] * len(img_paths))
			else:
				img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
				gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
				img_paths.sort()
				gt_paths.sort()
				# pdb.set_trace()
				img_tot_paths.extend(img_paths)
				gt_tot_paths.extend(gt_paths)
				tot_labels.extend([1] * len(img_paths))
				tot_types.extend([defect_type] * len(img_paths))
		
		assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

		return img_tot_paths, gt_tot_paths, tot_labels, tot_types

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, idx):
		img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
		img = Image.open(img_path).convert('RGB')
		img = self.transform(img)
		if gt == 0:
			gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
		else:
			gt = Image.open(gt)
			gt = self.gt_transform(gt)

		assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

		return img, gt, label, img_type


def int_parameter(level, maxval):
	return int(level * maxval / 10)


def float_parameter(level, maxval):
	return float(level) * maxval / 10.


def sample_level(n):
	return np.random.uniform(low=0.1, high=n)	


def autocontrast(pil_img, _):
	return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
	return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
	level = int_parameter(sample_level(level), 4)
	return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
	degrees = int_parameter(sample_level(level), 30)
	if np.random.uniform() > 0.5:
		degrees = -degrees
	return pil_img.rotate(degrees, resample=Image.Resampling.BILINEAR)


def solarize(pil_img, level):
	level = int_parameter(sample_level(level), 256)
	return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
	level = float_parameter(sample_level(level), 0.3)
	if np.random.uniform() > 0.5:
		level = -level
	return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
							Image.Transform.AFFINE, (1, level, 0, 0, 1, 0),
							resample=Image.Resampling.BILINEAR)


def shear_y(pil_img, level):
	level = float_parameter(sample_level(level), 0.3)
	if np.random.uniform() > 0.5:
		level = -level
	return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
							Image.Transform.AFFINE, (1, 0, 0, level, 1, 0),
							resample=Image.Resampling.BILINEAR)


def translate_x(pil_img, level):
	level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
	if np.random.random() > 0.5:
		level = -level
	return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
							Image.Transform.AFFINE, (1, 0, level, 0, 1, 0),
							resample=Image.Resampling.BILINEAR)


def translate_y(pil_img, level):
	level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
	if np.random.random() > 0.5:
		level = -level
	return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
							Image.Transform.AFFINE, (1, 0, 0, 0, 1, level),
							resample=Image.Resampling.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

def add_gaussian_noise(image, level=None):
	severity = 1
	image = np.array(image)
	corrupted = corrupt(image, corruption_name='gaussian_noise', severity=severity)
	image = Image.fromarray(corrupted)
	return image

def test_brightness(image, level=None):
	# severity = 1
	image = np.array(image)
	corrupted = corrupt(image, corruption_name='brightness', severity=level)
	image = Image.fromarray(corrupted)
	return image

def test_contrast(image, level=None):
	severity = 1
	image = np.array(image)
	# pdb.set_trace()
	corrupted = corrupt(image, corruption_name='contrast', severity=severity)
	image = Image.fromarray(corrupted)
	return image

def rotation(image, mask=None, level=0, fill_color=None):
	if fill_color is None:
		fill_color = (114, 114, 114)
	rotate_90 = True
	random_rotate = 30
	# pdb.set_trace()
	# rotate_90
	if rotate_90:
		degree = np.random.choice(np.array([0, 90, 180, 270]))
		image = image.rotate(
			degree, fillcolor=fill_color, resample=Image.BILINEAR
		)
		if mask is not None:
			mask = mask.rotate(degree, fillcolor=0, resample=Image.BILINEAR)
	# random_rotate
	if random_rotate > 0:
		degree = np.random.uniform(-random_rotate, random_rotate)
		image = image.rotate(
			degree, fillcolor=fill_color, resample=Image.BILINEAR
		)
		if mask is not None:
			mask = mask.rotate(degree, fillcolor=0, resample=Image.BILINEAR)
	if mask is not None:
		return image, mask
	else: 
		return image

def augmvtec(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):

	aug_list = [
		test_brightness, test_contrast, color, sharpness, add_gaussian_noise, add_gaussian_noise
	]
	# pdb.set_trace()
	severity = random.randint(1, severity)

	ws = np.float32(np.random.dirichlet([1] * width))
	m = np.float32(np.random.beta(alpha, alpha))
	preprocess_img = preprocess(image)
	mix = torch.zeros_like(preprocess_img)
	for i in range(width):
		image_aug = image.copy()
		depth = depth if depth > 0 else np.random.randint(
			1, 4)
		for _ in range(depth):
			op = np.random.choice(aug_list)
			image_aug = op(image_aug, level=severity)
		# Preprocessing commutes since all coefficients are convex
		mix += ws[i] * preprocess(image_aug)

	mixed = (1 - m) * preprocess_img + m * mix
	
	return mixed

def augmvtec2(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):

	aug_list = [
		test_brightness, test_contrast, add_gaussian_noise, rotation, rotation, rotation
	]
	# pdb.set_trace()
	severity = random.randint(1, severity)
	for i in range(width):
		image_aug = image.copy()
		depth = depth if depth > 0 else np.random.randint(
			1, 4)
		for _ in range(depth):
			op = np.random.choice(aug_list)
			image_aug = op(image_aug, level=severity)
			if op.__name__ == "rotation":
				break
		# Preprocessing commutes since all coefficients are convex
	mixed = preprocess(image_aug)
	
	return mixed

def augmvtec3(image, mask, preprocess, severity=3, width=3, depth=-1, alpha=1.):
	aug_list = [
		test_brightness, test_contrast, add_gaussian_noise, rotation, rotation, rotation
	]
	# aug_list = [
	# 	rotation, rotation, rotation
	# ]
	# pdb.set_trace()
	severity = random.randint(1, severity)
	# for i in range(width):
	image_aug = image.copy()
	depth = depth if depth > 0 else np.random.randint(
		1, 4)
	for _ in range(depth+width):
		op = np.random.choice(aug_list)
		if op.__name__ == "rotation":
			image_aug, mask = op(image_aug, mask, level=severity)
			break
		else:
			image_aug = op(image_aug, severity)
	# Preprocessing commutes since all coefficients are convex
	mixed = preprocess(image_aug)

	return mixed, mask

def augmvtec_guass(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):
	severity = 1
	image = np.array(image)
	corrupted = corrupt(image, corruption_name='gaussian_noise', severity=severity)
	image = Image.fromarray(corrupted)
	return preprocess(image)


class AugMixDatasetMVTec(torch.utils.data.Dataset):
	"""Dataset wrapper to perform AugMix augmentation."""

	def __init__(self, dataset, preprocess):
		self.dataset = dataset
		self.preprocess = preprocess
		self.gray_preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_train,
									std=std_train),
			transforms.Grayscale(3)
		])
	def __getitem__(self, i):
		x, _ = self.dataset[i]
		# pdb.set_trace()
		x = Image.fromarray(np.uint8(np.array(x)))
		# return self.preprocess(x), augmvtec(x, self.preprocess), self.gray_preprocess(x)
		if random.random() > 0.5:
			return self.preprocess(x), augmvtec(x, self.preprocess), augmvtec_guass(x, self.preprocess)
		else:
			return self.preprocess(x), augmvtec2(x, self.preprocess), augmvtec_guass(x, self.preprocess)

	def __len__(self):
		return len(self.dataset)

  
class SynAugMixDatasetMVTec(torch.utils.data.Dataset):
	"""Dataset wrapper to perform AugMix augmentation."""

	def __init__(self, root, preprocess, dtd_root):
		self.resize_transform = transforms.Compose([ transforms.Resize((256,256))])
		self.preprocess = preprocess
		self.gray_preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean_train,
									std=std_train),
			transforms.Grayscale(3)
		])
		self.img_paths = []
		rgb_paths = glob.glob(os.path.join(root, 'good') + "/*.png")
		rgb_paths.sort()
		self.img_paths.extend(rgb_paths)
		self.anomaly_source_paths = sorted(glob.glob(f"{dtd_root}/*/*.jpg"))

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, idx):
		rgb_path = self.img_paths[idx]
		# 原图
		img = Image.open(rgb_path).convert('RGB')
		x_ori = self.resize_transform(img)
		# 合成异常
		# img_file, cls_name = rgb_path.split('/')[-1], rgb_path.split('/')[-4]
		# fg_path = os.path.join(f'fg_mask/{cls_name}', img_file)
		# fg_mask = Image.open(fg_path)
		# fg_mask = np.asarray(fg_mask)[:, :, np.newaxis]  # [H, W, 1]
		# resized_depth_map = resize_organized_pc(fg_mask, img_size=256)
		# fore_mask = resized_depth_map > 0
		x = np.array(x_ori)
		x, mask, _ = transform_image(x, None, self.anomaly_source_paths)  # 不需要mask
		x = Image.fromarray(np.uint8(x))

		# return self.preprocess(x_ori), augmvtec(x, self.preprocess), mask
		if random.random() > 0.5:
			return self.preprocess(x_ori), augmvtec(x, self.preprocess), mask
		else:
			# pdb.set_trace()
			mask = Image.fromarray(np.uint8(mask[0].transpose(0,1)) * 255, mode='L')
			x_rot, mask = augmvtec3(x, mask, self.preprocess)
			# pdb.set_trace()
			mask = (np.array(mask)[None] / 255.0).round()
			return self.preprocess(x_ori), x_rot, mask


