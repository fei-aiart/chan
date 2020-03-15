#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test2.py
# @Author: Jehovah
# @Date  : 18-8-8
# @Desc  :


from torchvision.utils import save_image
from option import Options
from data_loader2 import *
from models.sys_trans_sp_multi import *
from pix2pix_model import *

opt = Options().parse()
net_G = Sys_Generator(opt.input_nc, opt.output_nc)
net_G.load_state_dict(torch.load(opt.pre_netG))

dataset = MyDataset(opt, isTrain=1)
data_iter = data.DataLoader(
    dataset, batch_size=opt.batchSize,
    num_workers=16)

# net_G.eval()
net_G.cuda()


for i, image in enumerate(data_iter):
    imgA = image[0]
    # imgB = image[1]
    # imgB = image['A']

    real_A = imgA.cuda()
    # real_B = imgB.cuda()

    fake_B = net_G(real_A)
    # output = output.cpu()
    output_name = '{:s}/{:s}{:s}'.format(
        opt.output, str(i+1), '.jpg')
    save_image(fake_B[:,:,3:253,28:228], output_name, normalize=True, scale_each=True)

    print output_name + "  succeed"

