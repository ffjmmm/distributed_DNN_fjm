from __future__ import print_function

import torch
import numpy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# from tensorboardX import SummaryWriter

import os
import argparse
import time

import vgg_new


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action="store_true", help='resume from checkpoint')
parser.add_argument('--original', action="store_true", help='use original VGG')
parser.add_argument('--dataset', type=str, default='ciffar10', help='choose the dataset')
parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
args = parser.parse_args()


# Data
def load_data():
    print('==> Preparing data..')
    time_data_start = time.time()
    if args.dataset == 'ciffar10':
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
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    else:
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
            transforms.Scale(384, Image.LANCZOS),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(enhance),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # validation data is already resized
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        train_loader = torchvision.datasets.ImageFolder('./data/' + 'train_no_resizing', train_transform)
        test_loader = torchvision.datasets.ImageFolder('./data/' + 'test', test_transform)

    time_data_end = time.time()
    print("Preparing data spends %fs\n" % (time_data_end - time_data_start))

    return train_loader, test_loader


# Training
def train(net, device, optimizer, criterion, epoch, train_loader, writer=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.print-freq == 0:
            print('Epoch: %d [%d/%d]: loss = %f, acc = %f' % (epoch, batch_idx, len(train_loader), loss.item(),
                                                              predicted.eq(targets).sum().item() / targets.size(0)))


def test(net, device, criterion, epoch, test_loader, best_acc, writer=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100. * correct / total
    print("%d : %f" % (epoch, acc))
    # writer.add_scalar('Acc', acc, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    return best_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Building model..')
    time_buildmodel_start = time.time()
    if args.original:
        print("use original VGG")
    else:
        net = vgg_new.VGG('VGG16')
    time_buildmodel_end = time.time()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    print("Building model spends %fs\n" % (time_buildmodel_end - time_buildmodel_start))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # writer = SummaryWriter('logs/distributed_DNN')
    train_loader, test_loader = load_data()
    for epoch in range(start_epoch, start_epoch+200):
        # train(net, device, optimizer, criterion, epoch, train_loader, writer)
        # best_acc = test(net, device, criterion, epoch, test_loader, best_acc, writer)
        train(net, device, optimizer, criterion, epoch, train_loader)
        best_acc = test(net, device, criterion, epoch, test_loader, best_acc)
    # writer.close()


if __name__ == '__main__':
    main()