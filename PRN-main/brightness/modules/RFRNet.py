import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.partialconv2d import PartialConv2d
from modules.Attention import AttentionModule
from torchvision import models

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=False)
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
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class RFRModule(nn.Module):
    def __init__(self, layer_size=6, in_channel = 64):
        super(RFRModule, self).__init__()
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        for i in range(3):
            name = 'enc_{:d}'.format(i + 1)
            out_channel = in_channel * 2
            block = [nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias = False),
                     nn.BatchNorm2d(out_channel),
                     nn.ReLU(inplace = True)]
            in_channel = out_channel
            setattr(self, name, nn.Sequential(*block))
        
        for i in range(3, 6):
            name = 'enc_{:d}'.format(i + 1)
            block = [nn.Conv2d(in_channel, out_channel, 3, 1, 2, dilation = 2, bias = False),
                     nn.BatchNorm2d(out_channel),
                     nn.ReLU(inplace = True)]
            setattr(self, name, nn.Sequential(*block))
        self.att = AttentionModule(512)
        for i in range(5, 3, -1):
            name = 'dec_{:d}'.format(i)
            block = [nn.Conv2d(in_channel + in_channel, in_channel, 3, 1, 2, dilation = 2, bias = False),
                     nn.BatchNorm2d(in_channel),
                     nn.LeakyReLU(0.2, inplace = True)]
            setattr(self, name, nn.Sequential(*block))
            

        block = [nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False),
                 nn.BatchNorm2d(512),
                 nn.LeakyReLU(0.2, inplace = True)]
        self.dec_3 = nn.Sequential(*block)
        
        block = [nn.ConvTranspose2d(768, 256, 4, 2, 1, bias = False),
                 nn.BatchNorm2d(256),
                 nn.LeakyReLU(0.2, inplace = True)]
        self.dec_2 = nn.Sequential(*block)
        
        block = [nn.ConvTranspose2d(384, 64, 4, 2, 1, bias = False),
                 nn.BatchNorm2d(64),
                 nn.LeakyReLU(0.2, inplace = True)]
        self.dec_1 = nn.Sequential(*block)
        
    def forward(self, input, mask):

        h_dict = {}  # for the output of enc_N

        h_dict['h_0']= input

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            h_key_prev = h_key
        
        h = h_dict[h_key]
        for i in range(self.layer_size - 1, 0, -1):
            enc_h_key = 'h_{:d}'.format(i)
            dec_l_key = 'dec_{:d}'.format(i)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)
            if i == 3:
                h = self.att(h, mask)
        return h

class RFRNet(nn.Module):
    def __init__(self):
        super(RFRNet, self).__init__()
        self.Pconv1 = PartialConv2d(4, 64, 7, 2, 3, multi_channel = True, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.Pconv2 = PartialConv2d(64, 64, 7, 1, 3, multi_channel = True, bias = False)
        self.bn20 = nn.BatchNorm2d(64)
        self.Pconv21 = PartialConv2d(64, 64, 7, 1, 3, multi_channel = True, bias = False)
        self.Pconv22 = PartialConv2d(64, 64, 7, 1, 3, multi_channel = True, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        self.RFRModule = RFRModule()


        self.conv31=nn.Conv2d(448,448,3,2,1,bias=False)
        self.bn31= nn.BatchNorm2d(448)
        self.conv32=nn.Conv2d(448,896,3,2,1,bias=False)
        self.bn32 = nn.BatchNorm2d(896)
        self.conv33=nn.Conv2d(896,896,3,2,1,bias=False)
        self.bn33 = nn.BatchNorm2d(896)
        self.conv34=nn.Conv2d(896,896,3,1,1,bias=False)
        self.bn34 = nn.BatchNorm2d(896)
        self.conv35=nn.Conv2d(896,896,3,1,1,bias=False)
        self.bn35 = nn.BatchNorm2d(896)
        self.conv36=nn.Conv2d(896,896,3,1,1,bias=False)
        self.bn36 = nn.BatchNorm2d(896)
        self.dconv37=nn.ConvTranspose2d(896, 896, 4, 2, 1, bias = False)
        self.bn37 = nn.BatchNorm2d(896)
        self.dconv38=nn.ConvTranspose2d(896, 448, 4, 2, 1, bias = False)
        self.bn38 = nn.BatchNorm2d(448)
        self.dconv39=nn.ConvTranspose2d(448, 256, 4, 2, 1, bias = False)
        self.bn39= nn.BatchNorm2d(256)
        self.conv40=nn.Conv2d(256, 128, 3, 1, 1, bias = False)
        self.bn40 = nn.BatchNorm2d(128)
        self.conv41=nn.Conv2d(128, 64, 3, 1, 1, bias = False)
        self.bn41 = nn.BatchNorm2d(64)


        self.Tconv = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tail1 = PartialConv2d(68, 32, 3, 1, 1, multi_channel = True, bias = False)
        self.tail2 = Bottleneck(32,8)
        self.out = nn.Conv2d(64,1,3,1,1, bias = False)

    def forward(self, in_image, mask):
        x1, m1 = self.Pconv1(in_image, mask)
        x1 = F.relu(self.bn1(x1), inplace = True)
        x1, m1 = self.Pconv2(x1, m1)
        x1 = F.relu(self.bn20(x1), inplace = True)
        x2 = x1
        x2, m2 = x1, m1
        n, c, h, w = x2.size()
        feature_group = [x2.view(n, c, 1, h, w)]
        mask_group = [m2.view(n, c, 1, h, w)]
        self.RFRModule.att.att.att_scores_prev = None
        self.RFRModule.att.att.masks_prev = None

        for i in range(6):
            x2, m2 = self.Pconv21(x2, m2)
            x2, m2 = self.Pconv22(x2, m2)
            x2 = F.leaky_relu(self.bn2(x2), inplace = True)
            x2 = self.RFRModule(x2, m2[:,0:1,:,:])
            x2 = x2 * m2
            feature_group.append(x2.view(n, c, 1, h, w))
            mask_group.append(m2.view(n, c, 1, h, w))
        x3 = torch.cat(feature_group, dim = 2)
        m3 = torch.cat(mask_group, dim = 2)


        # amp_vec = m3.mean(dim = 2)
        # print(x3.shape)                                         #torch.Size([1, 64, 7, 128, 128])
        # x3 = (x3*m3).mean(dim = 2) /(amp_vec+1e-7)
        # print(x3.shape)                                         #torch.Size([1, 64, 128, 128])
        # x3 = x3.view(n, c, h, w)
        # print(x3.shape)                                         #torch.Size([1, 64, 128, 128])
        # print(m3.shape)                                         #torch.Size([1, 64, 7, 128, 128])
        

        x3=x3*m3
        x3=x3.view(n,-1,128,128)
        x3=self.conv31(x3)
        x3=self.bn31(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.conv32(x3)
        x3=self.bn32(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.conv33(x3)
        x3=self.bn33(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.conv34(x3)
        x3=self.bn34(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.conv35(x3)
        x3=self.bn35(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.conv36(x3)
        x3=self.bn36(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.dconv37(x3)
        x3=self.bn37(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.dconv38(x3)
        x3=self.bn38(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.dconv39(x3)
        x3=self.bn39(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.conv40(x3)
        x3=self.bn40(x3)
        x3 = F.leaky_relu(x3, inplace = True)
        x3=self.conv41(x3)
        x3=self.bn41(x3)
        x3 = F.leaky_relu(x3, inplace = True)

        
        
        m3 = m3[:,:,-1,:,:]                                     
        #print(m3.shape)                                         #torch.Size([1, 64, 128, 128])
        x4 = self.Tconv(x3)
        x4 = F.leaky_relu(self.bn3(x4), inplace = True)
        m4 = F.interpolate(m3, scale_factor = 2)
        x5 = torch.cat([in_image, x4], dim = 1)
        m5 = torch.cat([mask, m4], dim = 1)
        x5, _ = self.tail1(x5, m5)
        x5 = F.leaky_relu(x5, inplace = True)
        x6 = self.tail2(x5)
        x6 = torch.cat([x5,x6], dim = 1)
        output = self.out(x6)
        return output, None
    
    def train(self, mode=True, finetune = False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()