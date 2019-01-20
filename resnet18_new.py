'''ResNet in PyTorch.
To call this function, use "net = resnet18_new.ResNet18_new(num_classes = 101)"
To print out the pkt size, uncomment line 103-126 and 295-301
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from input_activation import Lossy_Linear
from markov_random import markov_rand
import torch.backends.cudnn as cudnn
import numpy as np
import random
import time

global nonzero_pixels_rate
global bytes_per_packet
nonzero_pixels = []
bytes_per_packet = []

# new lossy_Conv2d without mask matrix
class lossy_Conv2d_new(nn.Module):
    def __init__(self, in_channels, out_channels, p11 = 0.99, p22 = 0.03, kernel_size=3, stride = 0, padding=0, num_pieces=(2, 2), bias=False):
        super(lossy_Conv2d_new, self).__init__()
        # for each pieces, define a new conv operation
        self.pieces = num_pieces
        self.p11 = p11
        self.p22 = p22
        self.stride = stride
        self.b1 = nn.Sequential(
            # use the parameters instead of numbers
            nn.Conv2d(in_channels, out_channels, stride = self.stride, kernel_size=kernel_size, padding=0, bias=False)
        )
        s1 = 112
        x1 = s1 // num_pieces[0] + 2
        y1 = s1 // num_pieces[1] + 2
            
        s2 = 56
        x2 = s2 // num_pieces[0] + 2
        y2 = s2 // num_pieces[1] + 2
        
        s3 = 28
        x3 = s3 // num_pieces[0] + 2
        y3 = s3 // num_pieces[1] + 2
        
        self.num_112 = 300
        self.mask_112 = markov_rand([16, self.num_112, x1, y1], p11, p22)
        self.num_56 = 500
        self.mask_56 = markov_rand([16, self.num_56, x2, y2], p11, p22)
        self.num_28 = 800
        self.mask_28 = markov_rand([16, self.num_28, x3, y3], p11, p22)

    def forward(self, x):
        # print("x shape : ", x.shape)
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
                    #print(xx.shape)
                    xx = F.pad(xx, (int(j == 0), int(j == pieces[1] - 1), int(i == 0), int(i == pieces[0] - 1), 0, 0, 0, 0))
                    xx = xx.cuda()
                    # print(xx.shape)
                    # mask = markov_rand(xx.shape, self.p11, self.p22)
                    
                    if dim[2] == 112:
                        self.mask_112 = self.mask_112.cuda()
                        offset = random.randint(0, self.num_112 - xx.shape[1] - 1)
                        mask = self.mask_112[0: dim[0], offset: offset + xx.shape[1], :, :]
                    elif dim[2] == 56:
                        self.mask_56 = self.mask_56.cuda()
                        offset = random.randint(0, self.num_56 - xx.shape[1] - 1)
                        mask = self.mask_56[0: dim[0], offset: offset + xx.shape[1], :, :]
                    elif dim[2] == 28:
                        self.mask_28 = self.mask_28.cuda()
                        offset = random.randint(0, self.num_28 - xx.shape[1] - 1)
                        mask = self.mask_28[0: dim[0], offset: offset + xx.shape[1], :, :]
                    
                    xx = xx * mask.cuda()                    
                    xx[:, :, 1: 1 + l_i, 1: 1 + l_j] = x[:, :, i * l_i: (i + 1) * l_i, j * l_j: (j + 1) * l_j]
                    dummy.append(xx.cuda())
                x_split.append(dummy)
            return x_split
        #print('x shape: ' + str(x.shape))
        x_split = split_dropout(x, self.pieces)
        #print('x_split' + str(x_split[0][0].shape))
        # time1 = time.time()
        r = []
        for i in range(self.pieces[0]):
            dummy = []
            for j in range(self.pieces[1]):
                #print('x_shape: ' + str(x_split[i][j].shape))
                rr = self.b1(x_split[i][j])
                #print('rr_shape: ' + str(rr.shape))
                dummy.append(rr)
            dummy_cat = torch.cat((dummy[0: self.pieces[1]]), 3)
            r.append(dummy_cat)    
        r = torch.cat((r[0: self.pieces[0]]), 2)

        #print('shape of r is ' + str(r.shape))
        # time2 = time.time()
        # print("Lossy Conv combine time = ", time2 - time1)
        return r.cuda()


class Quant_ReLU(nn.Module):
    def __init__(self, lower_bound=0.8, upper_bound=1, num_bits=4., num_pieces=(2, 2)):
        super(Quant_ReLU, self).__init__()
        # for each pieces, define a new conv operation
        self.num_bits = num_bits
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.delta = (upper_bound - lower_bound) / (2 ** self.num_bits - 1)
        self.num_pieces = num_pieces

    def forward(self, x):
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
        ###############################################################
        ###############recover this for printing out the pkt size######
        '''
        percent_nonzero_pixel = float(r1[r1>0].shape[0])/(x.shape[0]*x.shape[1]*(4*x.shape[2]-4))
        nonzero_pixel = float(r1[r1>0].shape[0])/(x.shape[0])
        byte_per_pkt = float(r1[r1>0].shape[0])/(x.shape[0])*4./(8*8) 
        total_length = r1.shape[1] * (2*(r1.shape[2] + r1.shape[3])-4)
        r2 = r1.cpu().detach().numpy()
        gap_length = 8192
        s = []
        for i in range(r1.shape[1]):
            s = s + r2[0,i,0,:].tolist()
            s = s + r2[0,i,r1.shape[2]-1,:].tolist()
            s = s + r2[0,i,:,0].tolist()
            s = s + r2[0,i,:,r1.shape[3]-1].tolist()  
        counter = 0;
        posi_num = 0;
        for i in range(len(s)):
            if((s[i] >= 0.001) or (counter == gap_length)):
                counter = 0
                posi_num = posi_num + 1;
            counter = counter + 1;
        nonzero_pixels.append(byte_per_pkt)    
        byte_per_pkt = byte_per_pkt + posi_num * np.log2(np.float(gap_length))/8
        bytes_per_packet.append(byte_per_pkt)        
        '''
        ##########################################################
        # quantize the pixels on the margin
        r1 = torch.round(r1 / self.delta) * self.delta
        # applies different mask to the pixel in the middle and on the margin
        r = F.relu(x * (1 - mask)) + r1
        return r

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, quant_relu_end=0, num_pieces = (2,2)):
        super(BasicBlock, self).__init__()
        self.quant_relu_end = quant_relu_end
        self.num_pieces = num_pieces
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu_quant = Quant_ReLU(lower_bound=.8, upper_bound=1, num_bits=4., num_pieces=self.num_pieces)
        self.shortcut = nn.Sequential()
        #self.expansion always equals 1
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if(self.quant_relu_end == 1):
            out = self.relu_quant(out)
        else:        
            out = F.relu(out)
        return out

class BasicBlock_lossy(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p11 = 0.99, p22 = 0.03, num_pieces=(2,2), lower_bound = 2, upper_bound = 3):
        super(BasicBlock_lossy, self).__init__()
        self.p11 = p11
        self.p22 = p22
        self.num_pieces = num_pieces
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.conv1 = lossy_Conv2d_new(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, p11 = self.p11, p22 = self.p22, num_pieces = self.num_pieces)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = lossy_Conv2d_new(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, p11 = self.p11, p22 = self.p22, num_pieces = self.num_pieces)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu_quant = Quant_ReLU(lower_bound=self.lower_bound, upper_bound=self.upper_bound, num_bits=4., num_pieces=self.num_pieces)
        self.shortcut = nn.Sequential()
        #self.expansion always equals 1
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # Since the filter size is one by one, do not have to use the lossy conv
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu_quant(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu_quant(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, block_lossy, num_blocks, num_classes=10, p11 = 0.99, p22 = 0.03, num_pieces= (2,2), loss_prob = 0.05, lower_bound = 2, upper_bound = 3, original = False):
        super(ResNet, self).__init__()
        self.original = original
        self.p11 = p11
        self.p22 = p22
        self.num_pieces = num_pieces
        self.in_planes = 64
        self.loss_prob = loss_prob
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, quant_relu_end = 0, num_pieces = num_pieces)
        if original:
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, quant_relu_end=0, num_pieces = num_pieces)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, quant_relu_end=0, num_pieces=num_pieces)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, quant_relu_end=0, num_pieces=num_pieces)
        else:
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, quant_relu_end = 1, num_pieces = num_pieces)
            self.layer3 = self._make_layers_lossy_conv(block_lossy = block_lossy, planes = 256, num_blocks = num_blocks[2], stride=2, p11 = self.p11 , p22 = self.p22, num_pieces = self.num_pieces)
            self.layer4 = self._make_layers_lossy_conv(block_lossy = block_lossy, planes = 512, num_blocks = num_blocks[3], stride=2, p11 = self.p11 , p22 = self.p22, num_pieces = self.num_pieces)
        self.linear = nn.Linear(512, num_classes)
        # self.linear = Lossy_Linear(512, num_classes, loss_prob=self.loss_prob)

    def _make_layer(self, block, planes, num_blocks, stride, quant_relu_end, num_pieces):
        strides = [stride] + [1]*(num_blocks-1)
        #print(strides)
        #sprint(strides)  #[1,1], [2,1], [2,1], [1,1]
        layers = []
        for i in range(len(strides)-1):
            #print(strides[i])
            layers.append(block(self.in_planes, planes, strides[i], quant_relu_end = 0, num_pieces = num_pieces))
            self.in_planes = planes * block.expansion
        #print(strides[-1])    
        #print(quant_relu_end)
        layers.append(block(self.in_planes, planes, strides[-1], quant_relu_end = quant_relu_end, num_pieces = num_pieces))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layers_lossy_conv(self, block_lossy, planes, num_blocks, stride, p11, p22, num_pieces):
        strides = [stride] + [1]*(num_blocks-1)
        #sprint(strides)  #[1,1], [2,1], [2,1], [1,1]
        layers = []
        for stride in strides:
            layers.append(block_lossy(self.in_planes, planes, stride, p11 = p11, p22 = p22, num_pieces = num_pieces))
            self.in_planes = planes * block_lossy.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.original:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
        else:
            # split the input matrix
            
            x_split = []
            xx = torch.chunk(x, self.f12_pieces[0], 2)
            for i in range(self.f12_pieces[0]):
                xxx = torch.chunk(xx[i], self.f12_pieces[1], 3)
                x_split.append(xxx)
                
            out = []
            for i in range(self.f12_pieces[0]):
                dummy = []
                for j in range(self.f12_pieces[1]):
                    rr = F.relu(self.bn1(self.conv1(x_split[i][j].cuda())))
                    rr = self.layer1(rr.cuda())
                    rr = self.layer2(rr.cuda())
                    dummy.append(rr)
                dummy_cat = torch.cat((dummy[0: self.f12_pieces[1]]), 3)
                out.append(dummy_cat)
            out = torch.cat((out[0: self.f12_pieces[0]]), 2)
            out.cuda()
            
            '''
            (x1,x2) = torch.chunk(x,2,2)
            (x11,x12) = torch.chunk(x1,2,3)
            (x21,x22) = torch.chunk(x2,2,3)

            #time1 = time.time()
            out11 = F.relu(self.bn1(self.conv1(x11)))
            out12 = F.relu(self.bn1(self.conv1(x12)))
            out21 = F.relu(self.bn1(self.conv1(x21)))
            out22 = F.relu(self.bn1(self.conv1(x22)))
            #time2 = time.time()
            #print("conv1 time : ", time2 - time1)
        
            # time1 = time.time()
            out11 = self.layer1(out11)   #([8, 64, 224, 224])
            out12 = self.layer1(out12)   #([8, 64, 224, 224])
            out21 = self.layer1(out21)   #([8, 64, 224, 224])
            out22 = self.layer1(out22)   #([8, 64, 224, 224])
            out11 = self.layer2(out11)   #([8, 128, 112, 112])
            out12 = self.layer2(out12)   #([8, 128, 112, 112])
            out21 = self.layer2(out21)   #([8, 128, 112, 112])
            out22 = self.layer2(out22)   #([8, 128, 112, 112])
            #time2 = time.time()
            #print("layer1&2 time : ", time2 - time1)
            
            out1 = torch.cat((out11,out12),3)
            out2 = torch.cat((out21,out22),3)    
            out = torch.cat((out1,out2),2)
            '''
        
        #time1 = time.time()
        out = self.layer3(out)   #([8, 256, 56, 56])
        out = self.layer4(out)   #([8, 512, 28, 28])
        #time2 = time.time()
        #print("layer3&4 time : ", time2 - time1)
        
        #time1 = time.time()
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #time2 = time.time()
        #print("linear time : ", time2 - time1)
        ###############################################################
        ###############recover this for printing out the pkt size######
        '''
        print('statistics is: ')
        avg1 = np.average(np.asarray(bytes_per_packet))
        avg2 = np.average(np.asarray(nonzero_pixels))
        print('bytes per pkt is: ' + str(avg1))
        print('number of nonzero pixels is ' + str(avg2))
        '''
        ###############################################################
        return out


def ResNet18_new(num_classes = 101, p11 = 0.99, p22 = 0.03, loss_prob = 0.1, num_pieces = (2,2), lower_bound = 2, upper_bound = 3, original=False):
    return ResNet(BasicBlock, BasicBlock_lossy, [2,2,2,2], num_classes = num_classes, p11 = 0.99, p22 = 0.03, loss_prob = 0.1, num_pieces = num_pieces, lower_bound = lower_bound, upper_bound = upper_bound, original=original)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18_new(num_classes=1000)
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    x = torch.randn(1, 3, 224, 224)
    y = net(x.cuda())
    print(y.size())

# test()