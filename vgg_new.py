#coding=utf-8

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from einops import rearrange, reduce

import time

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
    def __init__(self, in_channels, out_channels, alpha, kernel_size=3, padding=1, num_pieces=(2, 2)):
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

        
        def split_dropout(x, pieces=(2, 2), index_i=0, index_j=0):
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

            alpha = 0.5
            if index_i == 0 and index_j == 0:
                x_split[:, :, l_i + 1, 1: l_j + 2] = F.dropout(x[:, :, i_e, 0: j_e + 1], alpha, True) * (1 - alpha)
                x_split[:, :, 1: l_i + 1, l_j + 1] = F.dropout(x[:, :, 0: i_e, j_e], alpha, True) * (1 - alpha)
            if index_i == 0 and index_j == 1:
                x_split[:, :, l_i + 1, 0: l_j + 1] = F.dropout(x[:, :, i_e, j_s - 1: j_e], alpha, True) * (1 - alpha)
                x_split[:, :, 1: l_i + 1, 0] = F.dropout(x[:, :, 0: i_e, j_s - 1], alpha, True) * (1 - alpha)
            if index_i == 1 and index_j == 0:
                x_split[:, :, 0, 1: l_j + 2] = F.dropout(x[:, :, i_s - 1, 0: j_e + 1], alpha, True) * (1 - alpha)
                x_split[:, :, 1: l_i + 1, l_j + 1] = F.dropout(x[:, :, i_s: i_e, j_e], alpha, True) * (1 - alpha)
            if index_i == 1 and index_j == 1:
                x_split[:, :, 0, 0: l_j + 1] = F.dropout(x[:, :, i_s - 1, j_s - 1: j_e], alpha, True) * (1 - alpha)
                x_split[:, :, 1: l_i + 1, 0] = F.dropout(x[:, :, i_s: i_e, j_s - 1], alpha, True) * (1 - alpha)

            return x_split.cuda()
        
        def split_2(x, pieces):
            
            dim = x.shape

            x11 = x[:, :, 0: dim[2] // pieces[0] + 1, 0: dim[3] // pieces[1] + 1]
            x12 = x[:, :, 0: dim[2] // pieces[0] + 1, dim[3] // pieces[1] - 1: dim[3]]
            x21 = x[:, :, dim[2] // pieces[0] - 1: dim[2], 0: dim[3] // pieces[1] + 1]
            x22 = x[:, :, dim[2] // pieces[0] - 1: dim[2], dim[3] // pieces[1] - 1: dim[3]]

            '''
            x11 = torch.empty((dim[0], dim[1], dim[2] // pieces[0] + 1, dim[3] // pieces[1] + 1))
            x12 = torch.empty((dim[0], dim[1], dim[2] // pieces[0] + 1, dim[3] // pieces[1] + 1))
            x21 = torch.empty((dim[0], dim[1], dim[2] // pieces[0] + 1, dim[3] // pieces[1] + 1))
            x22 = torch.empty((dim[0], dim[1], dim[2] // pieces[0] + 1, dim[3] // pieces[1] + 1))
            '''
            '''
            x11 = F.pad(x11, (1, 1, 1, 1, 0, 0, 0, 0))
            x12 = F.pad(x12, (1, 1, 1, 1, 0, 0, 0, 0))
            x21 = F.pad(x21, (1, 1, 1, 1, 0, 0, 0, 0))
            x22 = F.pad(x22, (1, 1, 1, 1, 0, 0, 0, 0))
            '''

            x11 = F.pad(x11, (1, 0, 1, 0, 0, 0, 0, 0))
            x12 = F.pad(x12, (0, 1, 1, 0, 0, 0, 0, 0))
            x21 = F.pad(x21, (1, 0, 0, 1, 0, 0, 0, 0))
            x22 = F.pad(x22, (0, 1, 0, 1, 0, 0, 0, 0))

            x11 = x11.cuda()
            x12 = x12.cuda()
            x21 = x21.cuda()
            x22 = x22.cuda()

            '''
            x11.copy_(x[:, :, 0: dim[2] // pieces[0] + 1, 0: dim[3] // pieces[1] + 1])
            x12.copy_(x[:, :, 0: dim[2] // pieces[0] + 1, dim[3] // pieces[1] - 1: dim[3]])
            x21.copy_(x[:, :, dim[2] // pieces[0] - 1: dim[2], 0: dim[3] // pieces[1] + 1])
            x22.copy_(x[:, :, dim[2] // pieces[0] - 1: dim[2], dim[3] // pieces[1] - 1: dim[3]])
            '''

            alpha = self.alpha
            x11 = F.dropout(x11, p=alpha, training=True) * (1 - alpha)
            x12 = F.dropout(x12, p=alpha, training=True) * (1 - alpha)
            x21 = F.dropout(x21, p=alpha, training=True) * (1 - alpha)
            x22 = F.dropout(x22, p=alpha, training=True) * (1 - alpha)

            x11[:, :, 1: dim[2] // 2 + 1, 1: dim[3] // 2 + 1] = x[:, :, 0: dim[2] // pieces[0], 0: dim[3] // pieces[1]]
            x12[:, :, 1: dim[2] // 2 + 1, 1: dim[3] // 2 + 1] = x[:, :, 0: dim[2] // pieces[0], dim[3] // pieces[1]: dim[3]]
            x21[:, :, 1: dim[2] // 2 + 1, 1: dim[3] // 2 + 1] = x[:, :, dim[2] // pieces[0]: dim[2], 0: dim[3] // pieces[1]]
            x22[:, :, 1: dim[2] // 2 + 1, 1: dim[3] // 2 + 1] = x[:, :, dim[2] // pieces[0]: dim[2], dim[3] // pieces[1]: dim[3]]

            return x11.cuda(), x12.cuda(), x21.cuda(), x22.cuda()

        x11, x12, x21, x22 = split_2(x, self.pieces)

        '''
        x11 = split_rand(x, self.pieces, 0, 0)
        x12 = split_rand(x, self.pieces, 0, 1)
        x21 = split_rand(x, self.pieces, 1, 0)
        x22 = split_rand(x, self.pieces, 1, 1)
        '''
        '''
        x11 = split_dropout(x, self.pieces, 0, 0)
        x12 = split_dropout(x, self.pieces, 0, 1)
        x21 = split_dropout(x, self.pieces, 1, 0)
        x22 = split_dropout(x, self.pieces, 1, 1)
        '''

        # time1 = time.time()
        r11 = self.b1(x11)
        r12 = self.b1(x12)
        r21 = self.b1(x21)
        r22 = self.b1(x22)
        r1 = torch.cat((r11, r12), 3)
        r2 = torch.cat((r21, r22), 3)
        r_combine = torch.cat((r1, r2), 2)
        #time2 = time.time()
        #print("conv and combine time = ", time2 - time1)
        return r_combine


class lossy_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_pieces=(2, 2)):
        super(lossy_Conv2d, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        )

    def forward(self, x):
        # print(x.shape)
        def mask_matrix(dim=(16, 16, 4, 4), pieces=(2, 2), loss_prob=0.5):
            b_sub = torch.FloatTensor(dim[0], dim[1], dim[2] // pieces[1] + 2, dim[3] // pieces[0] + 2).uniform_() > 0.5
            # print(b_sub)
            b_sub_sub = torch.ones((dim[0], dim[1], dim[2] // pieces[1], dim[3] // pieces[0]))
            # print((dim[3],dim[2],dim[1],dim[0]))
            b_sub[:, :, 1:dim[2] // pieces[1] + 1, 1:dim[3] // pieces[0] + 1] = b_sub_sub

            # fold into the larger matrix
            mask_list = []
            for i in range(pieces[1]):
                dummy = []
                for j in range(pieces[0]):
                    b = torch.zeros((dim[0], dim[1], dim[2] + 2, dim[3] + 2))
                    b[:, :, i * dim[2] // pieces[1]:(i + 1) * dim[2] // pieces[1] + 2,
                    j * dim[3] // pieces[0]:(j + 1) * dim[3] // pieces[0] + 2] = b_sub

                    dummy.append(b[:, :, 1:-1, 1:-1].cuda())
                mask_list.append(dummy)
            return mask_list

        '''
        def one_mask_matrix(dim=(16, 16, 4, 4), pieces=(2, 2), loss_prob=0.5):
            # fold into the larger matrix
            mask_list = [];
            for i in range(pieces[1]):
                dummy = []
                for j in range(pieces[0]):
                    b = torch.zeros((dim[0], dim[1], dim[2], dim[3]))
                    b[:, :, i * dim[2] // pieces[1]:(i + 1) * dim[2] // pieces[1],
                    j * dim[3] // pieces[0]:(j + 1) * dim[3] // pieces[0]] = 1
                    dummy.append(b.cuda())
                mask_list.append(dummy)
            return mask_list
        '''
        mask = mask_matrix(x.shape, (2, 2), 0.5)
        # print(mask[0][0].shape)

        x11 = x * mask[0][0]
        x12 = x * mask[0][1]
        x21 = x * mask[1][0]
        x22 = x * mask[1][1]

        r11 = self.b1(x11)
        r12 = self.b1(x12)
        r21 = self.b1(x21)
        r22 = self.b1(x22)
        r1 = torch.cat((r11, r12), 3)
        r2 = torch.cat((r21, r22), 3)
        r = torch.cat((r1, r2), 2)

        '''
        one_mask = one_mask_matrix(r11.shape, (2, 2), 0.5)
        # concatenate the results
        r = r11 * one_mask[0][0] + r12 * one_mask[0][1] + r21 * one_mask[1][0] + r22 * one_mask[1][1]
        '''
        return r


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
    def __init__(self, vgg_name, dataset, original, alpha):
        super(VGG, self).__init__()
        # only accept VGG16
        self.features1 = self._make_layers(cfg['VGG16_1'], 3)
        self.features2 = self._make_layers(cfg['VGG16_2'], 64)
        if original:
            self.features3 = self._make_layers(cfg['VGG16_3'], 128)
            self.features4 = self._make_layers(cfg['VGG16_4'], 256)
        else :
            self.features3 = self._make_layers_lossy_conv(cfg['VGG16_3'], 128, alpha)
            self.features4 = self._make_layers_lossy_conv(cfg['VGG16_4'], 256, alpha)
        self.features5 = self._make_layers(cfg['VGG16_5'], 512)
        if dataset == 'CIFFAR10':
            self.classifier = nn.Linear(512, 10)
        else:
            self.classifier = nn.Linear(25088, 257)

    def forward(self, x):
        # split x
        # time1 = time.time()
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
        # time2 = time.time()
        # print("Time for feature 1 and 2: ", time2 - time1)

        # this is the end of the split

        # for feature 3, we have the loss transmission
        # mask = mask_matrix((out.shape[3],out.shape[2],out.shape[1],out.shape[0]),(2,2),0.5)
        # one_mask = one_mask_matrix((out.shape[3],out.shape[2],out.shape[1],out.shape[0]),(2,2),0.5)
        # time1 = time.time()
        out = self.features3(out)
        out = self.features4(out)
        # time2 = time.time()
        # print("Time for feature 3 and 4, loss conv: ", time2 - time1)

        # time1 = time.time()
        out = self.features5(out)
        # time2 = time.time()
        # print("Time for feature 5: ", time2 - time1)

        # time1 = time.time()
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # time2 = time.time()
        # print("Time for flatten and classify: ", time2 - time1)
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

    def _make_layers_lossy_conv(self, cfg, in_channels, alpha, relu_change=0):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [lossy_Conv2d_new(in_channels, x, kernel_size=3, padding=1, alpha=alpha),
                           nn.BatchNorm2d(x, affine=False),
                           Quant_ReLU(lower_bound=.5, upper_bound=.8, num_bits=4.)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG16', 'Caltech256')
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    x = torch.randn(64, 3, 224, 224)
    y = net(x)


# test()
