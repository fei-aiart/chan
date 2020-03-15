#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : option.py
# @Author: Jehovah
# @Date  : 18-6-4
# @Desc  : 

import os
import torch
import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser = argparse.ArgumentParser(description="PyTorch")
        self.parser.add_argument('--dataroot', default='/data/xxx/photosketch/',
                                 help="path to images (should have sub folders, eg: AR, CUHK etc)")
        self.parser.add_argument('--gpuid', type=str, default='0', help='which gpu to use')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
        self.parser.add_argument('--bata', type=int, default=0.5, help='momentum parameters bata1')
        self.parser.add_argument('--batchSize', type=int, default=1,
                                 help='with batchSize=1 equivalent to instance normalization.')
        self.parser.add_argument('--niter', type=int, default=800, help='number of epochs to train for')
        self.parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
        self.parser.add_argument('--sample', type=str, default='./samples', help='intermediate results are saved here')
        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints', help=' models are saved here')
        self.parser.add_argument('--output', default='./output', help='folder to output images ')
        self.parser.add_argument('--datalist', default='files/list_train.txt', help='use a text to load dataset and you\
                                 also need switch list when you test')
        self.parser.add_argument('--pre_netG', default='./checkpoints/net_G_ins.pth', help='load the pre-train model\
                                 and in train and load the final model in test')
        self.parser.add_argument('--pre_netD', default='./checkpoints/net_D_ins.pth', help=' ')
        self.parser.add_argument('--pre_netA', default='./checkpoints/net_A_ins.pth', help=' ')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        str_ids = opt.gpuid.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        mkdirs(opt.output)
        mkdirs(opt.sample)
        # save to the disk
        expr_dir = opt.checkpoints
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt = opt
        return self.opt


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
