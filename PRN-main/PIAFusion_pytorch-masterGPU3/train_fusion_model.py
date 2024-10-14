# -*- coding: utf-8 -*-
"""
@Time ： 2022/6/21 18:00
@Auth ： zxc (https://github.com/linklist2)
@File ：train_fusion_model.py
@IDE ：PyCharm
@Function ：训练融合网络
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.msrs_data import MSRS_data
from models.cls_model import Illumination_classifier
from models.common import gradient, clamp
from models.fusion_model import PIAFusion
import torch.nn as nn
from torchvision import models


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def l1_loss(f1, f2, mask=1):
    return torch.mean(torch.abs(f1 - f2) * mask)

def style_l( A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
    return loss_value

def TV_loss( x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
    w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
    return h_tv + w_tv

def preceptual_l(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/msrs_train',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='pretrained')  # 模型存储路径
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
    parser.add_argument('--loss_weight', default='[3, 7, 50]', type=str,
                        metavar='N', help='loss weight')
    parser.add_argument('--pretrained_module', default='pretrained/best_cls.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    train_dataset = MSRS_data(args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        model = PIAFusion()
        model = model.cuda()

        # 加载预训练的分类模型
        # one-hot标签[白天概率，夜晚概率]
        # cls_model = Illumination_classifier(input_channels=3)
        # cls_model.load_state_dict(torch.load(args.cls_pretrained))
        # # cls_model = cls_model.cuda()
        # cls_model.eval()


        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        try:
            model.load_state_dict(torch.load(args.pretrained_module))
        except FileNotFoundError:
            print("no pre module")


        for epoch in range(args.start_epoch, args.epochs):
            if epoch < args.epochs // 2:
                lr = args.lr
            else:
                lr = 0.5*args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

            # 修改学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            for bright_image, cartoon_image, mask, gt_image ,name in train_tqdm:
                mask=1-mask
                bright_image=bright_image.cuda()
                cartoon_image=cartoon_image.cuda()
                mask=mask.cuda()
                gt_image=gt_image.cuda()

                # vis_y_image = vis_y_image.cuda()
                # vis_image = vis_image.cuda()
                # inf_image = inf_image.cuda()
                optimizer.zero_grad()
                masked_img = gt_image * mask
                # print(bright_image.shape,cartoon_image.shape,mask.shape,gt_image.shape)

                vis_y_image = torch.cat((bright_image,masked_img),1)
                inf_image = torch.cat((cartoon_image,masked_img),1)

                fused_image = model(vis_y_image, inf_image)
                # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
                fused_image = clamp(fused_image)

                comp_image = fused_image*(1-mask)+masked_img

                # 使用预训练的分类模型，得到可见光图片属于白天还是夜晚的概率
                # pred = cls_model(vis_image)
                # day_p = pred[:, 0]
                # night_p = pred[:, 1]
                # vis_weight = day_p / (day_p + night_p)
                # inf_weight = 1 - vis_weight

                # # pixel l1_loss
                # loss_illum = F.l1_loss(inf_weight[:, None, None, None] * fused_image,
                #                        inf_weight[:, None, None, None] * inf_image) + F.l1_loss(
                #     vis_weight[:, None, None, None] * fused_image,
                #     vis_weight[:, None, None, None] * vis_y_image)
                #
                # # auxiliary intensity loss
                # loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))
                #
                # # gradient loss
                # gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_y_image)))
                # t1, t2, t3 = eval(args.loss_weight)
                # loss = t1 * loss_illum + t2 * loss_aux + t3 * gradinet_loss
                #

                lossNet = VGG16FeatureExtractor()
                lossNet=lossNet.cuda()
                a = lossNet(gt_image)
                b = lossNet(fused_image)
                c = lossNet(comp_image)

                real_B_feats = a
                fake_B_feats = b
                comp_B_feats = c

                tv_loss = TV_loss(comp_image * (1 - mask))
                style_loss = style_l(real_B_feats, fake_B_feats) + style_l(real_B_feats, comp_B_feats)
                preceptual_loss = preceptual_l(real_B_feats, fake_B_feats) + preceptual_l(real_B_feats,comp_B_feats)
                valid_loss = l1_loss(gt_image, fused_image, mask)
                hole_loss = l1_loss(gt_image, fused_image, (1 - mask))
                lone_loss = l1_loss(gt_image, fused_image)

                loss = (tv_loss * 0.1
                          + style_loss * 120
                          + preceptual_loss * 0.05
                          + valid_loss * 1
                          + hole_loss * 6)


                # loss =lone_loss


                # train_tqdm.set_postfix(epoch=epoch, tv_loss=(tv_loss * 0.1).item, style_loss=(style_loss * 120).item(),
                #                        preceptual_loss=(preceptual_loss * 0.05).item(),valid_loss=(valid_loss * 1).item(),
                #                        hole_loss=(hole_loss * 6).item,
                #                        loss_total=loss.item())
                train_tqdm.set_postfix(epoch=epoch,
                                       loss_total=loss.item(),st=style_loss.item(),pr=preceptual_loss.item(),va=valid_loss.item(),ho=hole_loss.item())





                loss.backward()
                optimizer.step()

            torch.save(model.state_dict(), f'{args.save_path}/fusion_model_epoch_{epoch}.pth')
