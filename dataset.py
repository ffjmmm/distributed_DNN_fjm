import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import time
import os

import torchvision
import torchvision.transforms as transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        f = open(txt, 'r')
        imgs = []
        for line in f:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            img = loader(words[0])
            imgs.append((img, int(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    
def get_CIFFAR10(root='./data', batch_size=256, num_workers=16):
    print('==> Preparing CIFFAR10 data..')
    time_data_start = time.time()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    time_data_end = time.time()
    print("Preparing data spends %fs\n" % (time_data_end - time_data_start))
    
    return train_loader, test_loader


def get_Caltech101(root='./data', batch_size=256, num_workers=32):
    print('==> Preparing Caltech101 data..')
    time_data_start = time.time()

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    factors = {
        0: lambda: np.random.normal(1.0, 0.3),
        1: lambda: np.random.normal(1.0, 0.1),
        2: lambda: np.random.normal(1.0, 0.1),
        3: lambda: np.random.normal(1.0, 0.3),
    }

    # random enhancers in random order
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image

    # train data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
        
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
        
    train_set = MyDataset(txt=os.path.join(root, 'dataset-train-101.txt'), transform=train_transform)
    test_set = MyDataset(txt=os.path.join(root, 'dataset-test-101.txt'), transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    time_data_end = time.time()
    print("Preparing data spends %fs\n" % (time_data_end - time_data_start))
    
    return train_loader, test_loader


def get_Caltech256(root='./data', batch_size=256, num_workers=32):
    print('==> Preparing Caltech256 data..')
    time_data_start = time.time()

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    factors = {
        0: lambda: np.random.normal(1.0, 0.3),
        1: lambda: np.random.normal(1.0, 0.1),
        2: lambda: np.random.normal(1.0, 0.1),
        3: lambda: np.random.normal(1.0, 0.3),
    }

    # random enhancers in random order
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image

    # train data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
        
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
        
    train_set = MyDataset(txt=os.path.join(root, 'dataset-train.txt'), transform=train_transform)
    test_set = MyDataset(txt=os.path.join(root, 'dataset-test.txt'), transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    time_data_end = time.time()
    print("Preparing data spends %fs\n" % (time_data_end - time_data_start))
    
    return train_loader, test_loader


def load_data(dataset, batch_size):
    if dataset == 'CIFFAR10':
        return get_CIFFAR10(batch_size=batch_size)
    elif dataset == 'Caltech101':
        return get_Caltech101(batch_size=batch_size)
    elif dataset == 'Caltech256':
        return get_Caltech256(batch_size=batch_size)