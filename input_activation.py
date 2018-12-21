import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module


class Partition_Linear_row(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Partition_Linear_row, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ii = torch.max(self.weight,0)[1]
        bb = torch.zeros(self.weight.shape).cuda()
        bb[ii,torch.linspace(0,bb.shape[1]-1,steps = bb.shape[1]).numpy()] = 1
        #print('bb shape')
        #print(bb.shape)
        #print('input shape')
        #print(input.shape)
        #print(F.linear(input, bb, bias=None))
        #torch.set_printoptions(threshold=10000)
        #print(bb)
        
        return F.linear(input, bb.t(), bias=None)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Partition_Linear_col(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Partition_Linear_col, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ii = torch.max(self.weight,1)[1]
        bb = torch.zeros(self.weight.shape).cuda()
        bb[torch.linspace(0,bb.shape[1]-1,steps = bb.shape[1]).numpy(),ii] = 1
        #print('bb shape')
        #print(bb.shape)
        #print('input shape')
        #print(input.shape)
        #print(F.linear(input, bb, bias=None))
        #torch.set_printoptions(threshold=10000)
        #print(bb)

        return F.linear(bb, input.t(), bias=None)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Lossy_Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, pieces = 4, loss_prob = 0.1, bias=True):
        super(Lossy_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.pieces = pieces
        self.block_size_x = in_features // self.pieces
        self.block_size_x_mod = in_features % self.pieces
        self.block_size_y = out_features // self.pieces
        self.block_size_y_mod = out_features % self.pieces
        self.loss_prob = loss_prob
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # generate a random matrix
        r = torch.rand((self.pieces,self.pieces)) > self.loss_prob
        # then extend it to the block random
        # u = np.concatenate((np.repeat(r.numpy(),self.block_size_y,axis = 1),np.ones((self.pieces,1))),axis = 1)
        u = np.concatenate((np.repeat(r.numpy(),self.block_size_y,axis = 1), np.ones((self.pieces, self.block_size_y_mod))),axis = 1)
        mask = torch.tensor(np.repeat(u,self.block_size_x,axis = 0)).cuda()
        #print(self.weight.shape)
        #print(self.block_size_y)
        #print(self.block_size_x)
        return F.linear(input, self.weight*mask.float().t(), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )