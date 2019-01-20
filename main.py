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
import resnet18_new
import dataset


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
parser.add_argument('--p11', default=0.99, type=float, help='p11')
parser.add_argument('--p22', default=0.03, type=float, help='p22')
parser.add_argument('--pieces_x', default=2, type=int, help='pieces to split on x')
parser.add_argument('--pieces_y', default=2, type=int, help='pieces to split on y')
parser.add_argument('--lossyLinear', action='store_true', help='use Lossy Linear')
parser.add_argument('--loss_prob', default=0.1, type=float, help='loss prob in Lossy Linear')
parser.add_argument('--lower_bound', default=0.5, type=float, help='lower bound in Quant ReLU')
parser.add_argument('--upper_bound', default=1.0, type=float, help='upper bound in Quant ReLU')
args = parser.parse_args()


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
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # time3 = time.time()
        # move_time = time3-time2
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # time4 = time.time()
        # forward_time = time4 - time3
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # time5 = time.time()
        # backward_time = time5 - time4
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
        print("data_time: ", data_time, "move_time:", move_time, "batch_time: ", batch_time)
        print("forward_time: ", forward_time, "backward_time: ", backward_time)
        '''
        
        
def test(net, device, criterion, epoch, test_loader, best_acc, writer=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # nonzero_rate = np.zeros((6, 1), dtype=float)
    # bytes_per_packet = np.zeros((6, 1), dtype=float)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 0:
                vgg_new.set_print_flag(True)
                vgg_new.init_array()
            outputs = net(inputs)
            if batch_idx == 0:
                nonzero, bytes_per = vgg_new.get_array()
                vgg_new.set_print_flag(False)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    print("Epoch %d finish, test acc : %f, best add : %f" % (epoch, acc, best_acc))
    writer.add_scalar('Acc', acc, epoch)
    writer.add_scalar('Loss', test_loss, epoch)
    
    for i in range(6):
        writer.add_scalar('Nonzero_Rate_' + str(i + 1), nonzero[i], epoch)
        writer.add_scalar('Bytes_Per_Packet_' + str(i + 1), bytes_per[i], epoch)
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + args.dataset + '_ResNet18_original' + '.t7')
        best_acc = acc

    return best_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Building model..')
    time_buildmodel_start = time.time()
    # net = vgg_new.VGG('VGG16', args.dataset, args.original, args.p11, args.p22, args.lossyLinear, args.loss_prob, (args.pieces_x, args.pieces_y), (args.pieces_x, args.pieces_y), args.lower_bound, args.upper_bound)
    net = resnet18_new.ResNet18_new(num_classes = 1000, lower_bound = args.lower_bound, upper_bound = args.upper_bound, num_pieces=(args.pieces_x, args.pieces_y), original=args.original)
    time_buildmodel_end = time.time()

    #net = net.to(device)
    net.cuda()
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

    name = 'Res18_'
    if args.original:
        name = name + 'Original_lr=' + str(args.lr) + '_bs=' + str(args.batch_size)
    else:
        name = name + 'Distributed_lr=' + str(args.lr) + '_Conv=(' + str(args.p11) + ',' + str(args.p22) + ')_ReLU=(' + str(args.lower_bound) + ',' + str(args.upper_bound) + ')_bs=' + str(args.batch_size) + '_' + str(args.pieces_x) + 'x' + str(args.pieces_y)
        if args.lossyLinear:
            name = name + '_lossyLinear=' + str(args.loss_prob)
        else:
            name = name + '_noLossyLinear'
    print(name)
    writer = SummaryWriter('logs/new_' + args.dataset + '/' + name)
    
    train_loader, test_loader = dataset.load_data(args.dataset, args.batch_size)
    
    print('==> Training..')
    for epoch in range(start_epoch, start_epoch + args.epoch):
        time_epoch_start = time.time()
        train(net, device, optimizer, criterion, epoch, train_loader)
        # best_acc = test(net, device, criterion, epoch, test_loader, best_acc)
        best_acc = test(net, device, criterion, epoch, test_loader, best_acc, writer)
        time_epoch_end = time.time()
        print("Epoch time : ", time_epoch_end - time_epoch_start)
    writer.close()


if __name__ == '__main__':
    main()
