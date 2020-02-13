import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_, zeros_
from utils.utils import *


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        conv1 = nn.Conv2d(in_channels=3,
                           out_channels=96,
                           kernel_size=(11,11),
                           stride=(2,2)
                           )
        bn1 = nn.BatchNorm2d(96)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(in_channels=96,
                          out_channels=256,
                          kernel_size=(5,5),
                          stride=(1,1))
        bn2 = nn.BatchNorm2d(256)
        conv3 = nn.Conv2d(in_channels=256,
                          out_channels=192,
                          kernel_size=(3,3),
                          stride=(1,1))
        bn3 = nn.BatchNorm2d(192)
        conv4 = nn.Conv2d(in_channels=192,
                          out_channels=192,
                          kernel_size=(3, 3),
                          stride=(1, 1))
        bn4 = nn.BatchNorm2d(192)
        conv5 = nn.Conv2d(in_channels=192,
                          out_channels=128,
                          kernel_size=(3,3),
                          stride=(1, 1))
        bn5 = nn.BatchNorm2d(128)

        pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        # self.embedding = nn.Sequential(conv1, pool1, conv2, pool2, conv3, conv4, conv5)
        self.embedding = nn.Sequential(conv1, bn1, relu, pool1,
                                       conv2, bn2, relu, pool2,
                                       conv3, bn3, relu,
                                       conv4, bn4, relu,
                                       conv5, bn5)

    def forward(self, input):
        embedding = self.embedding(input)
        return embedding


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        ########## Conv1 ##########
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1,
                        kernel_size=(3,3), stride=(1, 1),
                        padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        ########## Conv2 ##########
        ##########conv2_1##########
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                        kernel_size=(3, 3), stride=(2,2),
                        padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        ##########conv2_2##########
        self.conv2_2 = nn.Conv2d(in_channels=4, out_channels=2,
                                    kernel_size=(3,3), stride=(1,1),
                                    padding=1)
        self.conv2_2_relu = nn.ReLU()
        self.conv2_2_nin = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        ########## Conv3 ##########
        ##########conv3_1##########
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=2, groups=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        ##########conv3_mid##########
        self.conv3_mid = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1)

        ##########conv3_2##########
        self.conv3_2 = nn.Conv2d(in_channels=8, out_channels=4,
                                    kernel_size=3, padding=1, stride=1, groups=1)
        self.conv3_2_relu = nn.ReLU()
        self.conv3_2_nin = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        ##########conv3_3##########
        self.conv3_3 = nn.Conv2d(in_channels=8, out_channels=4,
                                    kernel_size=3, padding=1, stride=1, groups=1)
        self.conv3_3_relu = nn.ReLU()
        self.conv3_3_nin = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        ##########conv3_4##########
        self.conv3_4 = nn.Conv2d(in_channels=8, out_channels=4,
                                    kernel_size=3, padding=1, stride=1, groups=1)
        self.conv3_4_relu = nn.ReLU()
        self.conv3_4_nin = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(8)
        )

        ########## Conv4 ##########
        ##########conv4_1##########
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        ##########conv4_mid##########
        self.conv4_mid = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1)
        ##########conv4_2##########
        self.conv4_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        self.conv4_2_relu = nn.ReLU()
        self.conv4_2_nin = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        ##########conv4_3##########
        self.conv4_3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        self.conv4_3_relu = nn.ReLU()
        self.conv4_3_nin = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        ##########conv4_4##########
        self.conv4_4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        self.conv4_4_relu = nn.ReLU()
        self.conv4_4_nin = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
        )

        self.conv3_pooling = nn.AvgPool2d(4, 4)
        self.conv4_pooling = nn.AvgPool2d(2, 2)
        # examplar_featrue_size = (self.examplar_size[0]/16, self.examplar_size[1]/16)
        # cnadidate_feature_size = (self.candidate_size[0] / 16, self.candidate_size[1]/16)
        #
        # bias_size_h = (cnadidate_feature_size[0] -examplar_featrue_size[0]) + 1
        # bias_size_w = (cnadidate_feature_size[1] -examplar_featrue_size[1]) + 1
        #
        # self.b1 = 0.1 * torch.ones((batch_size, int(bias_size_h), int(bias_size_w)), dtype=torch.float32, requires_grad=True).to(device=self.device)

    def forward(self, x):
        out = self.forward_sep(x)
        return out

    def forward_sep(self, x):
        conv1 = self.conv1_1(x)
        conv2_1 = self.conv2_1(conv1)
        conv2_2 = self.c_relu_layer(conv2_1, self.conv2_2, self.conv2_2_relu, self.conv2_2_nin)
        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.c_relu_layer(conv3_1, self.conv3_2, self.conv3_2_relu, self.conv3_2_nin)
        conv3_3 = self.c_relu_layer(conv3_2, self.conv3_3, self.conv3_3_relu, self.conv3_3_nin)
        conv3_4 = self.c_relu_layer(conv3_3, self.conv3_4, self.conv3_4_relu, self.conv3_4_nin)
        conv3_mid = self.conv3_mid(conv3_1)
        elt3 = conv3_4*conv3_mid
        conv4_1 = self.conv4_1(elt3)
        conv4_2 = self.c_relu_layer(conv4_1, self.conv4_2, self.conv4_2_relu, self.conv4_2_nin)
        conv4_3 = self.c_relu_layer(conv4_2, self.conv4_3, self.conv4_3_relu, self.conv4_3_nin)
        conv4_4 = self.c_relu_layer(conv4_3, self.conv4_4, self.conv4_4_relu, self.conv4_4_nin)
        conv4_mid = self.conv4_mid(conv4_1)
        elt4 = conv4_4*conv4_mid

        conv3_pool = self.conv3_pooling(elt3)
        conv4_pool = self.conv4_pooling(elt4)
        ret = torch.cat((conv3_pool, conv4_pool), dim=1)
        return ret

    def c_relu_layer(self, value, conv_layer, relu_layer, nin_layer):
        p_val = conv_layer(value)
        n_val = p_val * -1
        concat = torch.cat((p_val, n_val), dim=1)
        concat = relu_layer(concat)
        ret = nin_layer(concat)
        return ret


class SiameseNet(nn.Module):
    def __init__(self, embedding_net, corr_func, score_sz, response_up):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.correl_func = SIM_DICT[corr_func]
        self.match_batchnorm = nn.BatchNorm2d(1)
        self.final_score_sz = response_up*(score_sz - 1) + 1

    def forward(self, *images):
        ref_embed = self.embedding_net(images[0])
        srch_embed = self.embedding_net(images[1])
        score_map = self.correl_func(ref_embed, srch_embed, self.match_batchnorm)
        score_map = F.interpolate(score_map, self.final_score_sz, mode='bicubic', align_corners=True)
        return score_map


def weight_init(model):
    if type(model) == nn.Conv2d:
        xavier_uniform_(model.weight, gain=math.sqrt(2.0))
        constant_(model.bias, 0.1)
    elif type(model) == nn.BatchNorm2d:
        normal_(model.weight, 1.0, 0.02)
        zeros_(model.bias)


FINAL_SIZE = {
    "Baseline": 17
}

import numpy as np
if __name__ == '__main__':
    gpu_state = torch.cuda.is_available()
    if gpu_state:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name()
        device = 'cuda'
    else:
        gpu_count = 0
        gpu_name = None
        device = 'cpu'
    a = np.random.rand(2,3,127, 127)
    b = np.random.rand(2,3,255, 255)
    m = Baseline().to(device)
    c = m(torch.from_numpy(a).float().to(device))

    # m = Model((a.shape[2], a.shape[3]), (b.shape[2], b.shape[3]), a.shape[0]).to(device=device)
    # correl = m(torch.from_numpy(a).float().to(device),torch.from_numpy(b).float().to(device))
    # a = 1


