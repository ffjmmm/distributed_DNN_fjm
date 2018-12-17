#coding=utf-8

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np

import time


num_gpu = 8
Quant_ReLU_rate = np.zeros((6, num_gpu), dtype=float)


def init_array():
    global Quant_ReLU_rate
    Quant_ReLU_rate = np.zeros((6, num_gpu), dtype=float)


def print_array():
    res = Quant_ReLU_rate.sum(axis=1)
    print(res)


def get_array():
    res = Quant_ReLU_rate.sum(axis=1) / 8.
    return res


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
    def __init__(self, in_channels, out_channels, alpha, kernel_size=3, padding=0, num_pieces=(2, 2)):
        super(lossy_Conv2d_new, self).__init__()

        # for each pieces, define a new conv operation
        self.pieces = num_pieces
        self.alpha = alpha
        self.b1 = nn.Sequential(
            # use the parameters instead of numbers
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        )

    def forward(self, x):
        # print("x shape : ", x.shape)

        def split_rand(x, pieces=(2, 2), index_i=0, index_j=0):
            dim = x.shape
            l_i = dim[2] // pieces[0]
            l_j = dim[3] // pieces[1]

            # [i_s:i_e, j_s:j_e]对应之前b_sub中间部分=1
            i_s = index_i * dim[2] // pieces[0]
            i_e = i_s + l_i
            j_s = index_j * dim[3] // pieces[1]
            j_e = j_s + l_j

            x_split = torch.zeros((dim[0], dim[1], l_i + 2, l_j + 2))
            x_split = x_split.cuda()
            x_split[:, :, 1: l_i + 1, 1: l_j + 1] = x[:, :, i_s: i_e, j_s: j_e]

            # generate random number to simulate the edge pixel loss
            alpha = 0.5
            if i_s > 0:
                rand = torch.FloatTensor(dim[0], dim[1], 1, l_j).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, 0, 1: l_j + 1] = x[:, :, i_s - 1, j_s: j_e] * rand[:, :, 0, :].cuda()
            if i_e < dim[2]:
                rand = torch.FloatTensor(dim[0], dim[1], 1, l_j).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, l_i + 1, 1: l_j + 1] = x[:, :, i_e, j_s: j_e] * rand[:, :, 0, :].cuda()
            if j_s > 0:
                rand = torch.FloatTensor(dim[0], dim[1], l_i, 1).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, 1: l_i + 1, 0] = x[:, :, i_s: i_e, j_s - 1] * rand[:, :, :, 0].cuda()
            if j_e < dim[3]:
                rand = torch.FloatTensor(dim[0], dim[1], l_i, 1).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, 1: l_i + 1, l_j + 1] = x[:, :, i_s: i_e, j_e] * rand[:, :, :, 0].cuda()

            # Four corner
            if i_s > 0 and j_s > 0:
                rand = torch.FloatTensor(dim[0], dim[1], 1, 1).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, 0, 0] = x[:, :, i_s - 1, j_s - 1] * rand[:, :, 0, 0].cuda()
            if i_e < dim[2] and j_s > 0:
                rand = torch.FloatTensor(dim[0], dim[1], 1, 1).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, l_i + 1, 0] = x[:, :, i_e, j_s - 1] * rand[:, :, 0, 0].cuda()
            if i_s > 0 and j_e < dim[3]:
                rand = torch.FloatTensor(dim[0], dim[1], 1, 1).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, 0, l_j + 1] = x[:, :, i_s - 1, j_e] * rand[:, :, 0, 0].cuda()
            if i_e < dim[2] and j_e < dim[3]:
                rand = torch.FloatTensor(dim[0], dim[1], 1, 1).uniform_() > alpha
                rand = rand.float()
                x_split[:, :, l_i + 1, l_j + 1] = x[:, :, i_e, j_e] * rand[:, :, 0, 0].cuda()

            return x_split.cuda()
        
        def split_dropout(x, pieces):
            
            dim = x.shape
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
                    
                    xx = F.dropout(xx, p=self.alpha, training=True) * (1 - self.alpha)
                    xx[:, :, 1: 1 + l_i, 1: 1 + l_j] = x[:, :, i * l_i: (i + 1) * l_i, j * l_j: (j + 1) * l_j]
                    # print(i, j, x_s, x_e, y_s, y_e, xx.shape)
                    dummy.append(xx.cuda())
                x_split.append(dummy)
                
            return x_split
        
        # time1 = time.time()
        x_split = split_dropout(x, self.pieces)
        # time2 = time.time()
        # print("Lossy Conv Split time = ", time2 - time1)
        
        '''
        x11 = split_rand(x, self.pieces, 0, 0)
        x12 = split_rand(x, self.pieces, 0, 1)
        x21 = split_rand(x, self.pieces, 1, 0)
        x22 = split_rand(x, self.pieces, 1, 1)
        '''
        
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
        # print(mask[0,0,:,:])
        '''
        x_mask = x * mask
        xx = x_mask > 0
        num_total = torch.sum(xx).cpu().numpy()
        xx1 = x_mask > self.lower_bound
        xx1 = xx1.cuda()
        xx2 = x_mask < self.upper_bound
        xx2 = xx2.cuda()
        xx = xx1 * xx2
        num_remain = torch.sum(xx).cpu().numpy()
        flag = True
        for i in range(6):
            for j in range(num_gpu):
                if Quant_ReLU_rate[i][j] == 0:
                    Quant_ReLU_rate[i][j] = num_remain / num_total
                    flag = False
                    break
            if flag == False:
                break
        if flag == True:
            print("ERROR!!!!!!")
        '''
        r1 = F.hardtanh(x * mask, self.lower_bound, self.upper_bound) - self.lower_bound
        # print(float(r1[r1>0].shape[0])/r1.view(-1).shape[0])
        # print(r1[0,0,:,:])
        # quantize the pixels on the margin
        r1 = torch.round(r1 / self.delta) * self.delta
        # print(r1[0,0,:,:])
        # applies different mask to the pixel in the middle and on the margin
        r = F.relu(x * (1 - mask)) + r1
        return r


class VGG(nn.Module):
    def __init__(self, vgg_name, dataset, original, alpha, pieces=(2, 2), f12_pieces=(2, 2)):
        super(VGG, self).__init__()
        # only accept VGG16
        self.f12_pieces = f12_pieces
        self.features1 = self._make_layers(cfg['VGG16_1'], 3)
        self.features2 = self._make_layers(cfg['VGG16_2'], 64)
        if original:
            self.features3 = self._make_layers(cfg['VGG16_3'], 128)
            self.features4 = self._make_layers(cfg['VGG16_4'], 256)
        else :
            self.features3 = self._make_layers_lossy_conv(cfg['VGG16_3'], 128, alpha, pieces)
            self.features4 = self._make_layers_lossy_conv(cfg['VGG16_4'], 256, alpha, pieces)
        self.features5 = self._make_layers(cfg['VGG16_5'], 512)
        if dataset == 'CIFFAR10':
            self.classifier = nn.Linear(512, 10)
        else:
            self.classifier = nn.Linear(25088, 257)

    def forward(self, x):
        # split x
        # print("input x: ", x.shape)
        '''
        out = self.features1(x)
        out = self.features2(out)
        '''
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
        
        '''
        (x1, x2) = torch.chunk(x, 2, 2)
        (x11, x12) = torch.chunk(x1, 2, 3)
        (x21, x22) = torch.chunk(x2, 2, 3)
        
        # split the input channel x
        out11 = self.features1(x11)
        out11 = self.features2(out11)
        out12 = self.features1(x12)
        out12 = self.features2(out12)
        out21 = self.features1(x21)
        out21 = self.features2(out21)
        out22 = self.features1(x22)
        out22 = self.features2(out22)
        out1 = torch.cat((out11, out12), 3)
        out2 = torch.cat((out21, out22), 3)
        out = torch.cat((out1, out2), 2)
        '''
        # this is the end of the split
        # for feature 3, we have the loss transmission
        # mask = mask_matrix((out.shape[3],out.shape[2],out.shape[1],out.shape[0]),(2,2),0.5)
        # one_mask = one_mask_matrix((out.shape[3],out.shape[2],out.shape[1],out.shape[0]),(2,2),0.5)
        # time1 = time.time()
        out = self.features3(out)
        out = self.features4(out)
        # time2 = time.time()
        # print("Feature 3 & 4 time = ", time2 - time1)
        
        out = self.features5(out)
        
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
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

    def _make_layers_lossy_conv(self, cfg, in_channels, alpha, pieces=(2, 2), relu_change=0):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [lossy_Conv2d_new(in_channels, x, kernel_size=3, padding=1, alpha=alpha, num_pieces=pieces),
                           nn.BatchNorm2d(x, affine=False),
                           Quant_ReLU(lower_bound=.5, upper_bound=.8, num_bits=4., num_pieces=pieces)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG16', 'Caltech256', False, 0.5, (2, 2), (2, 2))
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    x = torch.randn(64, 3, 224, 224)
    # x = torch.randn(128, 3, 32, 32)
    # init_array()
    y = net(x)
    # print_array()
    

# test()
