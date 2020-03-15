#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader2.py
# @Author: Jehovah
# @Date  : 18-7-30
# @Desc  : 



"""
load data
"""
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio


IMG_EXTEND = ['.jpg', '.JPG', '.jpeg', '.JPEG',
              '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
              ]


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTEND)


def mat_process(img_fl):
    """
    process mat, 11 channel to 8 channel
    :param img_fl:
    :return:
    """
    img_fl = img_fl.astype(np.float32)
    temp = img_fl
    lists = []
    refen = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 6), (7, 9), (8, 8), (10, 10)]
    for item in refen:
        aa, bb = item
        if aa == bb:
            ll = temp[aa, :, :]
        else:
            ll = temp[aa, :, :] + temp[bb, :, :]
            ll = np.where(ll > 1, 1, ll)
        lists.append(ll.reshape(1, ll.shape[0], ll.shape[1]))
    parsing = np.concatenate(lists, 0)

    return parsing


def make_dataset(dir, file):
    imgA = []
    imgB = []

    file = os.path.join(dir, file)
    fimg = open(file, 'r')
    for line in fimg:
        line = line.strip('\n')
        line = line.rstrip()
        word = line.split("||")
        imgA.append(os.path.join(dir, word[0]))
        imgB.append(os.path.join(dir, word[1]))

    return imgA, imgB


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(data.Dataset):
    def __init__(self, opt, isTrain=0, transform=None, return_paths=None, loader=default_loader):
        super(MyDataset, self).__init__()
        self.opt = opt
        self.isTrain = isTrain
        if isTrain == 0:
            datalist = self.opt.datalist
        else:
            datalist = self.opt.datalist.replace("train", "test")
        imgs = make_dataset(self.opt.dataroot, datalist)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + self.opt.dataroot + dir + "\n"
                                                                         "Supported image extensions are: " +
                                ",".join(IMG_EXTEND)))


        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path_A = self.imgs[0][index]
        path_B = self.imgs[1][index]

        imgA = Image.open(path_A).convert('RGB')
        if self.opt.output_nc == 3:
            imgB = Image.open(path_B).convert('RGB')
        else:
            imgB = Image.open(path_B).convert('L')
        if self.isTrain == 0:

            w, h = imgA.size
            pading_w = (self.opt.loadSize - w) / 2
            pading_h = (self.opt.loadSize - h) / 2
            padding = transforms.Pad((pading_w, pading_h), fill=0, padding_mode='constant')
            # padding = transforms.Pad((pading_w, pading_h), padding_mode='edge')
            i = random.randint(0, self.opt.loadSize - self.opt.fineSize)
            j = random.randint(0, self.opt.loadSize - self.opt.fineSize)

            imgA = self.process_img(imgA, i, j, padding)
            imgB = self.process_img(imgB, i, j, padding)


        else:

            w, h = imgA.size
            pading_w = (self.opt.fineSize - w) / 2
            pading_h = (self.opt.fineSize - h) / 2
            padding = transforms.Pad((pading_w, pading_h), fill=0, padding_mode='constant')
            # padding = transforms.Pad((pading_w, pading_h), padding_mode='edge')
            imgA = padding(imgA)
            imgB = padding(imgB)

            imgA = transforms.ToTensor()(imgA)
            imgB = transforms.ToTensor()(imgB)
            imgA = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgA)
            imgB = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgB)

        return imgA, imgB

    def __len__(self):
        return len(self.imgs[1])

    def process_img(self, img, i, j,padding):
        img = padding(img)
        img = img.crop((j, i, j + self.opt.fineSize, i + self.opt.fineSize))
        img = transforms.ToTensor()(img)
        # if self.isTrain == 0:
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        return img

    def process_parsing(self, mat_path, i, j, w, h):
        facelabel = sio.loadmat(mat_path)
        parsing = facelabel['res_label']
        parsing = np.transpose(parsing, (2, 1, 0))
        parsing = np.minimum(parsing, 1)
        parsing = np.maximum(parsing, 0)
        parsing = np.pad(parsing, ((0, 0), (w, w), (h, h)), 'edge')
        parsing = parsing[:, i:i+self.opt.fineSize, j:j+self.opt.fineSize]
        # parsing = np.where(parsing > 0.5, 1, 0)  # 二值化parsing

        parsing = parsing.astype('float32')
        torch_parsing = torch.from_numpy(parsing)

        return torch_parsing


if __name__ == '__main__':
    pass
