#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: Jehovah
# @Date  : 18-9-18
# @Desc  :


from torch.utils.data import DataLoader
import torchvision.utils as vutils
import time
from data_loader2 import *
from option import Options
from models.sys_trans_sp_multi import *
from pix2pix_model import *

opt = Options().parse()


def train():
    data_loader = MyDataset(opt)
    print('trainA images = %d' % len(data_loader))

    train_loader = torch.utils.data.DataLoader(dataset=data_loader, batch_size=opt.batchSize, shuffle=True,
                                               num_workers=16)

    test_set = MyDataset(opt, isTrain=1)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False,
                                              num_workers=16)
    print('testA images = %d' % len(test_loader))
    net_G = Sys_Generator(opt.input_nc, opt.output_nc)
    net_D = Discriminator(opt.input_nc, opt.output_nc)

    pre_dic_netG = torch.load(opt.pre_netG)
    low = net_G.state_dict()
    pre_dic_netG = {k: v for k, v in pre_dic_netG.items() if k in low}
    low.update(pre_dic_netG)
    net_G.load_state_dict(low)
    net_D.apply(weights_init)
    net_G.cuda()
    net_D.cuda()

    criterionGAN = GANLoss()
    critertion1 = nn.L1Loss()
    optimizerG = torch.optim.Adam(net_G.parameters(), lr=opt.lr, betas=(opt.bata, 0.999))
    optimizerD = torch.optim.Adam(net_D.parameters(), lr=opt.lr, betas=(opt.bata, 0.999))

    net_D.train()
    net_G.train()

    for epoch in range(1, opt.niter + 1):
        epoch_start_time = time.time()
        for i, image in enumerate(train_loader):
            imgA = image[0]
            imgB = image[1]

            real_A = imgA.cuda()
            fake_B = net_G(real_A)
            real_B = imgB.cuda()
            net_D.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = net_D(fake_AB.detach())

            loss_D_fake = criterionGAN(pred_fake, False)

            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = net_D(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizerD.step()

            # netG
            net_G.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            out_put = net_D(fake_AB)
            loss_G_GAN = criterionGAN(out_put, True)

            loss_G_L1 = critertion1(fake_B, real_B) * opt.lamb
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizerG.step()
            if i % 100 == 0:
                print ('[%d/%d][%d/%d] LOSS_D: %.4f LOSS_G: %.4f LOSS_L1: %.4f' % (
                    epoch, opt.niter, i, len(train_loader), loss_D, loss_G, loss_G_L1))
                print ('LOSS_real: %.4f LOSS_fake: %.4f' % (loss_D_real, loss_D_fake))
        print 'Time Taken: %d sec' % (time.time() - epoch_start_time)
        if epoch % 5 == 0:
            vutils.save_image(fake_B.data,
                              opt.sample + '/fake_samples_epoch_%03d.jpg' % (epoch),
                              normalize=True)
        if epoch >= 500:
            if epoch % 100 == 0:
                torch.save(net_G.state_dict(), opt.checkpoints + '/net_G_ins' + str(epoch) + '.pth')
                # torch.save(net_D.state_dict(), opt.checkpoints + '/net_D_ins' + str(epoch) + '.pth')
                print "saved model at epoch " + str(epoch)
    print "finish"


if __name__ == '__main__':
    train()
    pass
