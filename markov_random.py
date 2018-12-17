# genreate the random variable based on the two-state markov chain
import torch


def markov_rand(dim, p11 = 0.99, p22 = 0.03):
    p12 = 1- p11
    p21 = 1- p22
    state = 0
    filter_num = torch.ones((dim[0],4))
    # filter_num = filter_num.cuda()
    mask = torch.ones(dim)
    # mask = mask.cuda()
    trans_prob = torch.rand((dim[0],4))
    # trans_prob = trans_prob.cuda()
    for i in range(4):
        for j in range(dim[0]):
            if state == 0:
                filter_num[j][i] = 1
                if trans_prob[j][i] > p11:
                    state = 1
            elif state == 1:
                filter_num[j][i] = 0
                if trans_prob[j][i] > p22:
                    state = 0
    # generate the mask matrix
    for i in range(dim[0]):
        mask[i,:,0,:] =  filter_num[i][0]
    for i in range(dim[0]):
        mask[i,:,:,0] = filter_num[i][1]
    for i in range(dim[0]):
        mask[i,:,:,dim[3]-1] = filter_num[i][2]
    for i in range(dim[0]):
        mask[i,:,dim[2]-1,:] = filter_num[i][3]
    return mask
