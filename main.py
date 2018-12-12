from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance

import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

import os
import argparse
import time

import vgg_new


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', '-bs', default=256, type=int, help='set batch size')
parser.add_argument('--resume', '-r', action="store_true", help='resume from checkpoint')
parser.add_argument('--original', action="store_true", help='use original VGG')
parser.add_argument('--dataset', type=str, default='Caltech256', help='choose the dataset')
parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epoch', default=200, type=int, help='training epoch')
parser.add_argument('--alpha', default=0.4, type=float, help='edge pixels loss')
args = parser.parse_args()


# Data
def load_data():
    print('==> Preparing data..')
    time_data_start = time.time()
    if args.dataset != 'Caltech256':
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

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    else:
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

        '''
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_data = MyDataset(txt='./dataset-train.txt', transform=transform)
        test_data = MyDataset(txt='./dataset-test.txt', transform=transform)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=2)

        '''

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

        train_set = MyDataset(txt='./dataset-train.txt', transform=train_transform)
        test_set = MyDataset(txt='./dataset-test.txt', transform=test_transform)
        # train_set = torchvision.datasets.ImageFolder('./data/256_ObjectCategories/' + 'train', train_transform)
        # test_set = torchvision.datasets.ImageFolder('./data/256_ObjectCategories/' + 'test', test_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=32)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=32)

    time_data_end = time.time()
    print("Preparing data spends %fs\n" % (time_data_end - time_data_start))

    return train_loader, test_loader


# Training
def train(net, device, optimizer, criterion, epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    time1 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # time2 = time.time()
        # data_time = time2 - time1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        vgg_new.init_array()
        outputs = net(inputs)
        # vgg_new.print_array()
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.print_freq == 0:
            time2 = time.time()
            print('Epoch: %d [%d/%d]: loss = %f, acc = %f time = %d' % (epoch, batch_idx, len(train_loader), loss.item(), predicted.eq(targets).sum().item() / targets.size(0), time2 - time1))
            time1 = time.time()
        
        '''
        time1 = time.time()
        batch_time = time1 - time2
        print("data_time: ", data_time, "batch_time: ", batch_time)
        '''

def test(net, device, criterion, epoch, test_loader, best_acc, writer=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    Quant_ReLU_rate = np.zeros((6, 1), dtype=float)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print(batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            vgg_new.init_array()
            outputs = net(inputs)
            res = vgg_new.get_array()
            Quant_ReLU_rate = Quant_ReLU_rate + res
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    print("Epoch %d finish, test acc : %f, best add : %f" % (epoch, acc, best_acc))
    writer.add_scalar('Acc', acc, epoch)
    for i in range(6):
        writer.add_scalar('Quant_ReLU_Rate_' + str(i + 1), Quant_ReLU_rate[i], epoch)

    if epoch % 30 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + args.dataset + '.t7')

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + args.dataset + '.t7')
        best_acc = acc

    return best_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Building model..')
    time_buildmodel_start = time.time()
    net = vgg_new.VGG('VGG16', args.dataset, args.original, args.alpha)
    time_buildmodel_end = time.time()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_' + args.dataset + '.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    print("Building model spends %fs\n" % (time_buildmodel_end - time_buildmodel_start))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    name = 'VGG_'
    if args.original:
        name = name + 'Original_' + args.dataset + '_lr=' + str(args.lr) + '_bs=' + str(args.batch_size)
    else:
        name = name + 'Distributed_'
        name = name + args.dataset + '_lr=' + str(args.lr) + '_alpha=' + str(args.alpha) + '_bs=' + str(args.batch_size)
    print(name)
    # writer = SummaryWriter('logs/Caltech256/' + name)
    train_loader, test_loader = load_data()
    print('==> Training..')
    for epoch in range(start_epoch, start_epoch + args.epoch):
        time_epoch_start = time.time()
        train(net, device, optimizer, criterion, epoch, train_loader)
        best_acc = test(net, device, criterion, epoch, test_loader, best_acc)
        # best_acc = test(net, device, criterion, epoch, test_loader, best_acc, writer)
        time_epoch_end = time.time()
        print("Epoch time : ", time_epoch_end - time_epoch_start)
    # writer.close()


if __name__ == '__main__':
    main()
