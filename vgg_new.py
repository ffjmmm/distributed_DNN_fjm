#coding=utf-8

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
from input_activation import Lossy_Linear
from markov_random import markov_rand
import time


num_gpu = 8
nonzero_pixels_rate = np.zeros((6, num_gpu), dtype=float)
bytes_per_packet = np.zeros((6, num_gpu), dtype=float)


def init_array():
    global nonzero_pixels_rate
    global bytes_per_packet
    nonzero_pixels_rate = np.zeros((6, num_gpu), dtype=float)
    bytes_per_packet = np.zeros((6, num_gpu), dtype=float)

def get_array():
    nonzero = nonzero_pixels_rate.mean(axis=1)
    bytes_per = bytes_per_packet.mean(axis=1)
    return nonzero, bytes_per


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_1': [64, 64, 'M'],
    'VGG16_2': [128, 128, 'M'],
    'VGG16_3': [256, 256, 256, 'M'],
    'VGG16_4': [512, 512, 512, 'M'],
    'VGG16_5': [512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# new lossy_Conv2d without mask matrix
class lossy_Conv2d_new(nn.Module):
    def __init__(self, in_channels, out_channels, p11 = 0.99, p22 = 0.03, kernel_size=3, padding=0, num_pieces=(2, 2), dataset=None):
        super(lossy_Conv2d_new, self).__init__()
        # for each pieces, define a new conv operation
        self.pieces = num_pieces
        self.p11 = p11
        self.p22 = p22
        self.b1 = nn.Sequential(
            # use the parameters instead of numbers
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        )
        if dataset == 'Caltech256' or dataset == 'Caltech101':
            s1 = 56
            x1 = s1 // num_pieces[0] + 2
            y1 = s1 // num_pieces[1] + 2
            
            s2 = 28
            x2 = s2 // num_pieces[0] + 2
            y2 = s2 // num_pieces[1] + 2
            
            self.num_56 = 500
            self.mask_56 = markov_rand([32, self.num_56, x1, y1], p11, p22)
            self.num_28 = 800
            self.mask_28 = markov_rand([32, self.num_28, x2, y2], p11, p22)
            
    def forward(self, x):
        # print("x shape : ", x.shape)
        
        def split_dropout(x, pieces):
            
            dim = x.shape
            # print("input shape : ", dim)
            l_i = dim[2] // pieces[0]
            l_j = dim[3] // pieces[1]
            
            x_split = []
            for i in range(pieces[0]):
                dummy = []
                for j in range(pieces[1]):
                    x_s = 0 if i == 0 else i * l_i - 1
                    y_s = 0 if j == 0 else j * l_j - 1
                    x_e = (i + 1) * l_i if i == pieces[0] - 1 else (i + 1) * l_i + 1
                    y_e = (j + 1) * l_j if j == pieces[1] - 1 else (j + 1) * l_j + 1
                    xx = x[:, :, x_s: x_e, y_s: y_e]
                    xx = F.pad(xx, (int(j == 0), int(j == pieces[1] - 1), int(i == 0), int(i == pieces[0] - 1), 0, 0, 0, 0))
                    xx = xx.cuda()
                    # print(xx.shape)
                    # mask = markov_rand(xx.shape, self.p11, self.p22)
                    if dim[2] == 56:
                        self.mask_56 = self.mask_56.cuda()
                        offset = random.randint(0, self.num_56 - xx.shape[1] - 1)
                        mask = self.mask_56[0: dim[0], offset: offset + xx.shape[1], :, :]
                    else:
                        self.mask_28 = self.mask_28.cuda()
                        offset = random.randint(0, self.num_28 - xx.shape[1] - 1)
                        mask = self.mask_28[0: dim[0], offset: offset + xx.shape[1], :, :]
                    xx = xx * mask.cuda()
                    
                    # xx = F.dropout(xx, p=0.3, training=True) * (1 - 0.3)
                    xx[:, :, 1: 1 + l_i, 1: 1 + l_j] = x[:, :, i * l_i: (i + 1) * l_i, j * l_j: (j + 1) * l_j]
                    dummy.append(xx.cuda())
                x_split.append(dummy)
                
            return x_split
        
        # time1 = time.time()
        x_split = split_dropout(x, self.pieces)
        # time2 = time.time()
        # print("Lossy Conv Split time = ", time2 - time1)
        
        # time1 = time.time()
        r = []
        for i in range(self.pieces[0]):
            dummy = []
            for j in range(self.pieces[1]):
                rr = self.b1(x_split[i][j])
                dummy.append(rr)
            dummy_cat = torch.cat((dummy[0: self.pieces[1]]), 3)
            r.append(dummy_cat)    
        r = torch.cat((r[0: self.pieces[0]]), 2)
        # time2 = time.time()
        # print("Lossy Conv combine time = ", time2 - time1)
        return r.cuda()


class Quant_ReLU(nn.Module):
    def __init__(self, lower_bound=0.8, upper_bound=1., num_bits=4., num_pieces=(2, 2)):
        super(Quant_ReLU, self).__init__()
        # for each pieces, define a new conv operation
        self.num_bits = num_bits
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.delta = (upper_bound - lower_bound) / (2 ** self.num_bits - 1)
        self.num_pieces = num_pieces

    def forward(self, x):
        # print(x.shape)
        def gen_mask(dim=(16, 16, 4, 4), pieces=(2, 2)):
            mask = torch.zeros(dim[2], dim[3])
            for i in range(1, pieces[0]):
                mask[i * dim[2] // pieces[0] - 1, :] = 1;
                mask[i * dim[2] // pieces[0], :] = 1;
            for j in range(1, pieces[1]):
                mask[:, j * dim[3] // pieces[1] - 1] = 1;
                mask[:, j * dim[3] // pieces[1]] = 1;
            return mask.cuda()

        mask = gen_mask(x.shape, self.num_pieces)
        r1 = F.hardtanh(x * mask, self.lower_bound, self.upper_bound) - self.lower_bound
        
        flag = True
        for i in range(6):
            for j in range(num_gpu):
                if nonzero_pixels_rate[i][j] == 0:
                    nonzero_pixels_rate[i][j] = float(r1[r1>0].shape[0])/(x.shape[0]*x.shape[1]*(4*x.shape[2]-4))
                    bytes_per_packet[i][j] = float(r1[r1>0].shape[0])/(x.shape[0])*4./(8*8)
                    print(i, j, nonzero_pixels_rate[i][j], bytes_per_packet[i][j])
                    flag = False
                    break
            if flag == False:
                break
        if flag == True:
            print("ERROR!!!!!!")
        
        ################################################################
        # print("Ratio of nonzero pixels:" + str(float(r1[r1>0].shape[0])/(x.shape[0]*x.shape[1]*(4*x.shape[2]-4))))
        # print("number of bytes per packet:" + str(float(r1[r1>0].shape[0])/(x.shape[0])*4./(8*8)))
        ################################################################
        # print(float(r1[r1>0].shape[0])/r1.view(-1).shape[0])
        # print(r1[0,0,:,:])
        # quantize the pixels on the margin
        r1 = torch.round(r1 / self.delta) * self.delta
        # print(r1[0,0,:,:])
        # applies different mask to the pixel in the middle and on the margin
        r = F.relu(x * (1 - mask)) + r1
        return r


class VGG(nn.Module):
    def __init__(self, vgg_name, dataset, original, p11=0.99, p22=0.03, lossyLinear=False, loss_prob=0.1, pieces=(2, 2), f12_pieces=(2, 2), lower_bound=0.5, upper_bound=1.0):
        super(VGG, self).__init__()
        # only accept VGG16
        self.f12_pieces = f12_pieces
        self.features1 = self._make_layers(cfg['VGG16_1'], 3)
        self.features2 = self._make_layers(cfg['VGG16_2'], 64)
        self.original = original
        if original:
            self.features3 = self._make_layers(cfg['VGG16_3'], 128)
            self.features4 = self._make_layers(cfg['VGG16_4'], 256)
        else :
            self.features3 = self._make_layers_lossy_conv(cfg['VGG16_3'], 128, p11, p22, dataset, pieces, lower_bound=lower_bound, upper_bound=upper_bound)
            self.features4 = self._make_layers_lossy_conv(cfg['VGG16_4'], 256, p11, p22, dataset, pieces, lower_bound=lower_bound, upper_bound=upper_bound)
        self.features5 = self._make_layers(cfg['VGG16_5'], 512)
        if dataset == 'CIFFAR10':
            if lossyLinear:
                self.classifier = Lossy_Linear(512, 10, loss_prob=loss_prob)
            else:
                self.classifier = nn.Linear(512, 10)
        elif dataset == 'Caltech256':
            if lossyLinear:
                self.classifier = Lossy_Linear(25088, 257, loss_prob=loss_prob)
            else:
                self.classifier = nn.Linear(25088, 257)
        elif dataset == 'Caltech101':
            if lossyLinear:
                self.classifier = nn.Sequential(
                    Lossy_Linear(25088, 4096, loss_prob=loss_prob),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    Lossy_Linear(4096, 4096, loss_prob=loss_prob),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    Lossy_Linear(4096, 101, loss_prob=loss_prob)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(25088, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 101)
                )

    def forward(self, x):
        # split x
        # print("input x: ", x.shape)
        if self.original:
            out = self.features1(x)
            out = self.features2(out)
        else:
            x_split = []
            xx = torch.chunk(x, self.f12_pieces[0], 2)
            for i in range(self.f12_pieces[0]):
                xxx = torch.chunk(xx[i], self.f12_pieces[1], 3)
                x_split.append(xxx)
        
            # time1 = time.time()
            out = []
            for i in range(self.f12_pieces[0]):
                dummy = []
                for j in range(self.f12_pieces[1]):
                    rr = self.features1(x_split[i][j].cuda())
                    rr = self.features2(rr.cuda())
                    dummy.append(rr)
                dummy_cat = torch.cat((dummy[0: self.f12_pieces[1]]), 3)
                out.append(dummy_cat)    
            out = torch.cat((out[0: self.f12_pieces[0]]), 2)
            out.cuda()
        # time2 = time.time()
        # print("Feature1 & 2 time = ", time2 - time1)

        # this is the end of the split
        # for feature 3, we have the loss transmission
        time1 = time.time()
        out = self.features3(out)
        out = self.features4(out)
        time2 = time.time()
        # print("Feature 3 & 4 time = ", time2 - time1)
        
        out = self.features5(out)
        
        out = out.view(out.size(0), -1)
        # time1 = time.time()
        out = self.classifier(out)
        # time2 = time.time()
        # print("Linear time = ", time2 - time1)
        return out

    def _make_layers(self, cfg, in_channels, relu_change=0):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _make_layers_lossy_conv(self, cfg, in_channels, p11, p22, dataset, pieces=(2, 2), relu_change=0, lower_bound=0.5, upper_bound=1.0):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [lossy_Conv2d_new(in_channels, x, kernel_size=3, p11 = p11, p22 = p22, num_pieces=pieces, dataset=dataset),
                           nn.BatchNorm2d(x, affine=False),
                           Quant_ReLU(lower_bound=lower_bound, upper_bound=upper_bound, num_bits=4., num_pieces=pieces)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG16', 'Caltech256', False, lossyLinear=True)
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    x = torch.randn(256, 3, 224, 224)
    # x = torch.randn(128, 3, 32, 32)
    # init_array()
    y = net(x)
    # print_array()
    

test()
