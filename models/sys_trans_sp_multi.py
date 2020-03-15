#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-26 下午3:13
# @Author  : Jehovah
# @File    : systhesis.py
# @Software: PyCharm
import torch

import torch.nn as nn


class Sys_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(Sys_Generator, self).__init__()
        self.en_1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.en_2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.resblock = nn.Sequential(
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2)
        )
        self.resblock_2 = nn.Sequential(
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2)
        )
        self.resblock_1 = nn.Sequential(
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2)
        )

        self.resblock1 = ResidualBlock(in_channels=512, out_channels=512)
        self.resblock2 = ResidualBlock(in_channels=512, out_channels=512)
        self.resblock3 = ResidualBlock(in_channels=512, out_channels=512)

        self.en1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.en2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True)
        )
        self.en3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True)
        )

        self.en4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en6 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en7 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en8 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8,ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),

            nn.ReLU(True)
        )
        self.de6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.de7 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.de8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc,
                               kernel_size=4, stride=2,
                               padding=1),
            nn.Tanh()
        )

        self.de8_1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc,
                               kernel_size=4, stride=2,
                               padding=1),
            nn.Tanh()
        )


        # self.ta2 = Trans_Attn(ngf * 8)
        # self.ta3 = Trans_Attn(ngf * 8)
        # self.ta4 = Trans_Attn(ngf * 8)
        # self.ta5 = Trans_Attn(ngf * 4)
        self.ta6 = Trans_Attn(ngf * 2)
        self.sp = Spacial_Attn(ngf * 2)


    def forward(self, x):

        out_en1 = self.en1(x)
        out_en2 = self.en2(out_en1)
        out_en3 = self.en3(out_en2)
        out_en4 = self.en4(out_en3)
        out_en5 = self.en5(out_en4)
        out_en6 = self.en6(out_en5)
        out_en7 = self.en7(out_en6)
        out_en8 = self.en8(out_en7)
        out_en8 = self.resblock1(out_en8,is_bn=False)
        out_en8 = self.resblock2(out_en8,is_bn=False)
        out_en8 = self.resblock3(out_en8,is_bn=False)

        #decoder
        out_de1 = self.de1(out_en8)
        out_de1 = torch.cat((out_de1, out_en7), 1)
        out_de2 = self.de2(out_de1)
        # out_de2 = self.ta2(out_en6, out_de2)
        out_de2 = torch.cat((out_de2, out_en6), 1)
        out_de3 = self.de3(out_de2)
        # out_de3 = self.ta3(out_en5, out_de3)
        out_de3 = torch.cat((out_de3, out_en5), 1)
        out_de4 = self.de4(out_de3)
        # out_de4 = self.ta4(out_en4, out_de4)
        out_de4 = torch.cat((out_de4, out_en4), 1)
        out_de5 = self.de5(out_de4)
        # out_de5 = self.ta5(out_en3, out_de5)
        out_de5 = torch.cat((out_de5, out_en3), 1)
        out_de6 = self.de6(out_de5)
        out_de6 = self.ta6(out_en2, out_de6)
        out_de6 = torch.cat((out_de6, out_en2), 1)
        out_de7 = self.de7(out_de6)
        out_de7 = torch.cat((out_de7, out_en1), 1)
        # out_de8 = self.de8(out_de7)
        # out_2 = self.de8_1(out_de7)
        out_1 = self.en_1(x)
        out_1 = self.en_2(out_1)
        out_1 = self.resblock_2(out_1)

        out_1, out_de7 = self.sp(out_1, out_de7)

        out_1 = out_1+out_de7
        out_1 = self.resblock(out_1)
        out_1 = self.resblock_1(out_1)
        out_de8 = self.de8(out_1)


        return out_de8


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x, is_bn=True):
        residual = x
        out = self.conv1(x)
        if is_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if is_bn:
            out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.cov4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.cov5 = nn.Sequential(
            nn.Conv2d(ndf*8, ndf * 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.cov5_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(ndf * 8),
        )
        self.cov5_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.InstanceNorm2d(ndf * 4),
        )
        self.cov5_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),
            nn.InstanceNorm2d(ndf * 2),
        )


        self.cls = nn.Sequential(
            nn.Conv2d(1408, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_cov1 = self.cov1(x)
        out_cov2 = self.cov2(out_cov1)
        out_cov3 = self.cov3(out_cov2)
        out_cov4 = self.cov4(out_cov3)
        out_1 = self.cov5(out_cov4)
        out_2 = self.cov5_1(out_cov4)
        out_3 = self.cov5_2(out_cov3)
        out_4 = self.cov5_3(out_cov2)
        out = torch.cat((out_1, out_2, out_3, out_4), 1)
        out = self.cls(out)
        return out


class Spacial_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Spacial_Attn, self).__init__()
        self.chanel_in = in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim*2, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):

        xy = torch.cat((x, y), 1)

        out = self.conv(xy)

        y = y * out
        x = x*(1 - out)

        return x, y

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out



class Trans_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Trans_Attn, self).__init__()
        self.sa1 = Self_Attn(in_dim)
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)

        self.sa2 = Self_Attn(in_dim)

        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)

        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

        self.in1 = nn.InstanceNorm2d(in_dim)
        self.in2 = nn.InstanceNorm2d(in_dim)
        self.in3 = nn.InstanceNorm2d(in_dim)
        self.in4 = nn.InstanceNorm2d(in_dim)
        self.in5 = nn.InstanceNorm2d(in_dim)
        self.in6 = nn.InstanceNorm2d(in_dim)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        out_1 = self.sa1(x)
        out_1 = self.in1(out_1)
        out_2 = self.conv1(out_1)
        out_2 = self.in2(out_1+out_2)

        out_3 = self.sa2(y)
        out_3 = self.in3(out_3)

        m_batchsize, C, width, height = out_2.size()
        proj_query = self.query_conv(out_2).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(out_2).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(out_3).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + y
        # 归一化 out
        out = self.in4(out)
        # out = torch.nn.functional.normalize(out.view(m_batchsize,C, -1), 1).resize(m_batchsize, C, width, height)
        out = self.in5(out+out_3)
        out_4 = self.conv2(out)
        out = self.in6(out+out_4)
        return out





