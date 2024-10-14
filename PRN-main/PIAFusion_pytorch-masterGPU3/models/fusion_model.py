import torch
from torch import nn
from models.common import reflect_conv

class Gated_Conv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,rate=1,activation=nn.functional.relu ):
        super(Gated_Conv, self).__init__()
        padding=int(rate*(ksize-1)/2)
        #通过卷积将通道数变成输出两倍，其中一半用来做门控，学习
        self.conv=nn.Conv2d(in_ch,2*out_ch,kernel_size=ksize,stride=stride,padding=padding,dilation=rate)
        self.activation=activation
    def forward(self,x):
        raw=self.conv(x)
        x1=raw.split(int(raw.shape[1]/2),dim=1)#将特征图分成两半，其中一半是做学习
        gate=torch.sigmoid(x1[0])#将值限制在0-1之间
        out=self.activation(x1[1])
        out=out*gate
        return out


class Tran_Gated_Conv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,rate=1,activation=nn.functional.relu ):
        super(Tran_Gated_Conv, self).__init__()
        padding=int(rate*(ksize-1)/2)
        #通过卷积将通道数变成输出两倍，其中一半用来做门控，学习
        self.conv=torch.nn.ConvTranspose2d(in_ch,2*out_ch,kernel_size=ksize,stride=stride,padding=padding,output_padding=1,dilation=rate)
        self.activation=activation
    def forward(self,x):
        raw=self.conv(x)
        x1=raw.split(int(raw.shape[1]/2),dim=1)#将特征图分成两半，其中一半是做学习
        gate=torch.sigmoid(x1[0])#将值限制在0-1之间
        out=self.activation(x1[1])
        out=out*gate
        return out


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30

class Channel_Attention_Module_FC(nn.Module):
    def __init__(self, channels, ratio):
        super(Channel_Attention_Module_FC, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c)
        max_x = self.max_pooling(x).view(b, c)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1)
        return x * v


class DSatt(nn.Module):
    def __init__(self, k=7,p=3):
        super(DSatt, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()  
    def forward(self, feature):  # x.size() 30,40,50,30
        
        vi_feature, ir_feature=feature.split(int(feature.shape[1]/2),dim=1)
        sub_ir_vi = ir_feature - vi_feature
        sub_vi_ir = vi_feature - ir_feature

        avg_out = torch.mean(sub_ir_vi, dim=1, keepdim=True)
        max_out, _ = torch.max(sub_ir_vi, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        vi_feature=sub_ir_vi*self.sigmoid(x)

        avg_out = torch.mean(sub_vi_ir, dim=1, keepdim=True)
        max_out, _ = torch.max(sub_vi_ir, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv2(x)  # 30,1,50,30
        ir_feature=sub_vi_ir*self.sigmoid(x)



        return  vi_feature,ir_feature # 30,1,50,30  



def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    batch_size, channels, _, _ = vi_feature.size()

    sub_vi_ir = vi_feature - ir_feature
    vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))

    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))

    # 特征加上各自的带有简易通道注意力机制的互补特征
    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = Gated_Conv(in_ch=4,out_ch=16,ksize=5,stride=2,rate=1)
        self.ir_conv1 = Gated_Conv(in_ch=6,out_ch=16,ksize=5,stride=2,rate=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn11 = nn.BatchNorm2d(16)

        self.vi_conv2 = Gated_Conv(in_ch=16,out_ch=32,ksize=5,stride=2,rate=1)
        self.ir_conv2 = Gated_Conv(in_ch=16,out_ch=32,ksize=5,stride=2,rate=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn22 = nn.BatchNorm2d(32)

        self.vi_conv3 = Gated_Conv(in_ch=32,out_ch=64,ksize=3,stride=2,rate=1)
        self.ir_conv3 = Gated_Conv(in_ch=32,out_ch=64,ksize=3,stride=2,rate=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn33 = nn.BatchNorm2d(64)

        self.vi_conv4 = Gated_Conv(in_ch=64,out_ch=128,ksize=3,stride=1,rate=1)
        self.ir_conv4 = Gated_Conv(in_ch=64,out_ch=128,ksize=3,stride=1,rate=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn44 = nn.BatchNorm2d(128)


        self.vi_conv5 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.ir_conv5 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn55 = nn.BatchNorm2d(128)

        self.vi_conv51 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.ir_conv51 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.bn51 = nn.BatchNorm2d(128)
        self.bn551 = nn.BatchNorm2d(128)

        self.vi_conv52 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.ir_conv52 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.bn52 = nn.BatchNorm2d(128)
        self.bn552 = nn.BatchNorm2d(128)

        self.vi_conv53 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.ir_conv53 = Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=1,rate=1)
        self.bn53 = nn.BatchNorm2d(128)
        self.bn553 = nn.BatchNorm2d(128)

        self.vi_conv6 = Tran_Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=2,rate=1)
        self.ir_conv6 = Tran_Gated_Conv(in_ch=128,out_ch=128,ksize=3,stride=2,rate=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn66 = nn.BatchNorm2d(128)

        self.vi_conv7 = Tran_Gated_Conv(in_ch=128,out_ch=64,ksize=3,stride=2,rate=1)
        self.ir_conv7 = Tran_Gated_Conv(in_ch=128,out_ch=64,ksize=3,stride=2,rate=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn77= nn.BatchNorm2d(64)

        self.vi_conv8 = Tran_Gated_Conv(in_ch=64,out_ch=32,ksize=3,stride=2,rate=1)
        self.ir_conv8 = Tran_Gated_Conv(in_ch=64,out_ch=32,ksize=3,stride=2,rate=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn88 = nn.BatchNorm2d(32)

        self.vi_conv9 = Gated_Conv(in_ch=32,out_ch=32,ksize=3,stride=1,rate=1)
        self.ir_conv9 = Gated_Conv(in_ch=32,out_ch=32,ksize=3,stride=1,rate=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.bn99= nn.BatchNorm2d(32)

        self.DS1=DSatt()
        self.DS2=DSatt()
        self.DS3=DSatt()


        # self.vi_conv1 = nn.Conv2d(in_channels=4, kernel_size=1, out_channels=16, stride=1, padding=0)
        # self.ir_conv1 = nn.Conv2d(in_channels=6, kernel_size=1, out_channels=16, stride=1, padding=0)
        #
        # self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        # self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        #
        # self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        # self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        #
        # self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        # self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        #
        # self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        # self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out = activate(self.bn1(self.vi_conv1(y_vi_image)))
        ir_out = activate(self.bn11(self.ir_conv1(ir_image)))

        # vi_out, ir_out = CMDAF(activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))
        # vi_out, ir_out = CMDAF(activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))
        # vi_out, ir_out = CMDAF(activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))
        vi_out = activate(self.bn2(self.vi_conv2(vi_out)))
        ir_out = activate(self.bn22(self.ir_conv2(ir_out)))

        vi_out = activate(self.bn3(self.vi_conv3(vi_out)))
        ir_out = activate(self.bn33(self.ir_conv3(ir_out)))

        vi_out = activate(self.bn4(self.vi_conv4(vi_out)))
        ir_out = activate(self.bn44(self.ir_conv4(ir_out)))

        vi_out = activate(self.bn5(self.vi_conv5(vi_out)))
        ir_out = activate(self.bn55(self.ir_conv5(ir_out)))

        feature=Fusion(vi_out,ir_out)
        a,b=self.DS1(feature)
        vi_out=vi_out+a
        ir_out=ir_out+b

        vi_out = activate(self.bn51(self.vi_conv51(vi_out)))
        ir_out = activate(self.bn551(self.ir_conv51(ir_out)))

        vi_out = activate(self.bn52(self.vi_conv52(vi_out)))
        ir_out = activate(self.bn552(self.ir_conv52(ir_out)))

        vi_out = activate(self.bn53(self.vi_conv53(vi_out)))
        ir_out = activate(self.bn553(self.ir_conv53(ir_out)))

        feature=Fusion(vi_out,ir_out)
        a,b=self.DS2(feature)
        vi_out=vi_out+a
        ir_out=ir_out+b

        vi_out = activate(self.bn6(self.vi_conv6(vi_out)))
        ir_out = activate(self.bn66(self.ir_conv6(ir_out)))

        vi_out = activate(self.bn7(self.vi_conv7(vi_out)))
        ir_out = activate(self.bn77(self.ir_conv7(ir_out)))

        feature=Fusion(vi_out,ir_out)
        a,b=self.DS3(feature)
        vi_out=vi_out+a
        ir_out=ir_out+b

        vi_out = activate(self.bn8(self.vi_conv8(vi_out)))
        ir_out = activate(self.bn88(self.ir_conv8(ir_out)))






        vi_out, ir_out = activate(self.bn9(self.vi_conv9(vi_out))), self.bn99(activate(self.ir_conv9(ir_out)))
        return vi_out, ir_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.conv1 = Gated_Conv(in_ch=512,out_ch=256,ksize=3,stride=1,rate=1)
        # self.conv2 = Gated_Conv(in_ch=256,out_ch=128,ksize=3,stride=1,rate=1)
        # self.conv3 = Gated_Conv(in_ch=128,out_ch=64,ksize=3,stride=1,rate=1)
        # self.conv4 = Gated_Conv(in_ch=64,out_ch=32,ksize=3,stride=1,rate=1)
        # self.conv5 = Gated_Conv(in_ch=32,out_ch=3,ksize=3,stride=1,rate=1)
        self.ch_att1 = Channel_Attention_Module_FC(64, 64)

        self.conv1 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv11 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn11 = nn.BatchNorm2d(32)

        self.ch_att2 = Channel_Attention_Module_FC(32, 32)

        self.conv2 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv22 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn22 = nn.BatchNorm2d(32)

        self.ch_att3 = Channel_Attention_Module_FC(32, 32)


        self.conv3 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv33 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn33 = nn.BatchNorm2d(32)

        self.conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv44 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn44 = nn.BatchNorm2d(32)

        self.conv5 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv55 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn55 = nn.BatchNorm2d(32)

        self.conv6 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv66 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn66 = nn.BatchNorm2d(32)

        self.conv7 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(in_channels=32, kernel_size=3, out_channels=3, stride=1, padding=1)
        self.out=nn.Sigmoid()

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = x + self.ch_att1(x)
        x = activate(self.bn1(self.conv1(x)))
        x = activate(self.bn11(self.conv11(x)))
        x = x + self.ch_att2(x)
        x2 = x
        x = activate(self.bn2(self.conv2(x)))
        x = activate(self.bn22(self.conv22(x)))
        x = x+x2
        x = x + self.ch_att3(x)
        x3 = x
        x = activate(self.bn3(self.conv3(x)))
        x = activate(self.bn33(self.conv33(x)))
        x = x+x3

        x4 = x
        x = activate(self.bn4(self.conv4(x)))
        x = activate(self.bn44(self.conv44(x)))
        x = x+x4

        x5 = x
        x = activate(self.bn5(self.conv5(x)))
        x = activate(self.bn55(self.conv55(x)))
        x = x+x5

        x6 = x
        x = activate(self.bn6(self.conv6(x)))
        x = activate(self.bn66(self.conv66(x)))
        x = x+x6


        x = self.bn7(self.conv7(x))
        x = self.conv8(x)
        x = self.out(x)
        # x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


class PIAFusion(nn.Module):
    def __init__(self):
        super(PIAFusion, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, y_vi_image, ir_image):
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)
        encoder_out = Fusion(vi_encoder_out, ir_encoder_out)
        fused_image = self.decoder(encoder_out)
        return fused_image
