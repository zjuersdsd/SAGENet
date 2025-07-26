# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 12:33
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

    def forward(self, x):
        weights = torch.zeros(self.num_channels, 1, self.stride, self.stride)
        if torch.cuda.is_available():
            weights = weights.cuda()
        weights[:, :, 0, 0] = 1
        return F.conv_transpose2d(x, weights, stride=self.stride, groups=self.num_channels)


class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                (module_name, nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size,
                                                 stride, padding, output_padding, bias=False)),
                ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
                ('relu', nn.ReLU(inplace=True)),
            ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))


class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
            ('unpool', Unpool(in_channels)),
            ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
            ('relu', nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels // 2)
        self.layer3 = self.upconv_module(in_channels // 4)
        self.layer4 = self.upconv_module(in_channels // 8)


class FasterUpConv(Decoder):
    # Faster Upconv using pixelshuffle

    class faster_upconv_module(nn.Module):

        def __init__(self, in_channel):
            super(FasterUpConv.faster_upconv_module, self).__init__()

            self.conv1_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=3)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv2_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(2, 3))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv3_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(3, 2))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv4_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=2)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.ps = nn.PixelShuffle(2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            # print('Upmodule x size = ', x.size())
            x1 = self.conv1_(nn.functional.pad(x, (1, 1, 1, 1)))
            x2 = self.conv2_(nn.functional.pad(x, (1, 1, 0, 1)))
            x3 = self.conv3_(nn.functional.pad(x, (0, 1, 1, 1)))
            x4 = self.conv4_(nn.functional.pad(x, (0, 1, 0, 1)))

            x = torch.cat((x1, x2, x3, x4), dim=1)

            output = self.ps(x)
            output = self.relu(output)

            return output

    def __init__(self, in_channel):
        super(FasterUpConv, self).__init__()

        self.layer1 = self.faster_upconv_module(in_channel)
        self.layer2 = self.faster_upconv_module(in_channel // 2)
        self.layer3 = self.faster_upconv_module(in_channel // 4)
        self.layer4 = self.faster_upconv_module(in_channel // 8)


class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels // 2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm1', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU()),
                ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels // 2)
        self.layer3 = self.UpProjModule(in_channels // 4)
        self.layer4 = self.UpProjModule(in_channels // 8)


class FasterUpProj(Decoder):
    # Faster UpProj decorder using pixelshuffle

    class faster_upconv(nn.Module):

        def __init__(self, in_channel):
            super(FasterUpProj.faster_upconv, self).__init__()

            self.conv1_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=3)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv2_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(2, 3))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv3_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(3, 2))),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.conv4_ = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=2)),
                ('bn1', nn.BatchNorm2d(in_channel // 2)),
            ]))

            self.ps = nn.PixelShuffle(2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            # print('Upmodule x size = ', x.size())
            x1 = self.conv1_(nn.functional.pad(x, (1, 1, 1, 1)))
            x2 = self.conv2_(nn.functional.pad(x, (1, 1, 0, 1)))
            x3 = self.conv3_(nn.functional.pad(x, (0, 1, 1, 1)))
            x4 = self.conv4_(nn.functional.pad(x, (0, 1, 0, 1)))
            # print(x1.size(), x2.size(), x3.size(), x4.size())

            x = torch.cat((x1, x2, x3, x4), dim=1)

            x = self.ps(x)
            return x

    class FasterUpProjModule(nn.Module):
        def __init__(self, in_channels):
            super(FasterUpProj.FasterUpProjModule, self).__init__()
            out_channels = in_channels // 2

            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('faster_upconv', FasterUpProj.faster_upconv(in_channels)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = FasterUpProj.faster_upconv(in_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channel):
        super(FasterUpProj, self).__init__()

        self.layer1 = self.FasterUpProjModule(in_channel)
        self.layer2 = self.FasterUpProjModule(in_channel // 2)
        self.layer3 = self.FasterUpProjModule(in_channel // 4)
        self.layer4 = self.FasterUpProjModule(in_channel // 8)


def choose_decoder(decoder, in_channels):
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    elif decoder == "fasterupproj":
        return FasterUpProj(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.view(b, c, 1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y

class ChirpMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChirpMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class SqueezeExciteFusion(nn.Module):
    def __init__(self, chirp_input_dim, audio_feature_dim, fusion_dim):
        super(SqueezeExciteFusion, self).__init__()
        self.chirp_mlp = ChirpMLP(chirp_input_dim, fusion_dim)
        self.se_block = SEBlock(fusion_dim)
        self.audio_fc = nn.Linear(audio_feature_dim, fusion_dim)

    def forward(self, chirp_params, audio_features):
        chirp_features = self.chirp_mlp(chirp_params)
        audio_features = self.audio_fc(audio_features)
        fused_features = chirp_features + audio_features
        output = self.se_block(fused_features)
        return output

class FCRN_encoder(nn.Module):
    def __init__(self, layers = 50, output_feature_dim=512, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(FCRN_encoder, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048


        self.resize_conv = nn.Conv2d(2, 3, kernel_size=1)
        self.resize_bn = nn.BatchNorm2d(3)
        # self.resize_conv = SpatiallySeparableConv2d(2, 3, kernel_size=1)
        # self.resize_bn = nn.BatchNorm2d(3)
        # self.resize_conv = nn.Conv2d(2, 3, kernel_size=3, padding=2, dilation=2)
        # self.resize_bn = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)

        self.conv_reduce = nn.Conv2d(num_channels // 2, num_channels // 4, kernel_size=(9, 6), stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(num_channels // 4)

        self.conv_output = nn.Conv2d(num_channels // 4, output_feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_output = nn.BatchNorm2d(output_feature_dim)

        # self.upSample = choose_decoder(decoder, num_channels // 2)

        # # setting bias=true doesn't improve accuracy
        # self.conv3 = nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)

        # self.upSample.apply(weights_init)

        # self.conv3.apply(weights_init)

    def forward(self, x):
        # resize
        x = self.resize_conv(x)
        x = self.resize_bn(x)
        # print('x size = ', x.size())

        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv2(x4)
        x = self.bn2(x)

        # print('x size = ', x.size())
        x = self.conv_reduce(x)
        x = self.bn_reduce(x)
        x = self.conv_output(x)
        x = self.bn_output(x)

        return x

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net layer whose learning rate is 1x lr.
        """
        b = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters of the net layer whose learning rate is 20x lr.
        """
        b = [self.conv2, self.bn2, self.upSample, self.conv3, self.bilinear]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

from torch.nn import TransformerEncoder, TransformerEncoderLayer
class Angular_encoder(nn.Module):
    def __init__(self, feature_dim=512, num_layers=2, nhead=4, dim_feedforward=512, dropout=0.1):
        super(Angular_encoder, self).__init__()
        self.feature_dim = feature_dim
        self.input_dim = 181
        self.embedding = nn.Linear(self.input_dim, feature_dim)
        encoder_layers = TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

# class AngularQEncoder(nn.Module):
#     def __init__(self, q_init_dim=2, d_query=512, num_heads=8):
#         super(AngularQEncoder, self).__init__()
#         self.q_init_dim = q_init_dim
#         self.d_query = d_query
#         self.num_heads = num_heads
        
#         # 线性变换层
#         self.query_proj = nn.Sequential(
#             nn.Linear(q_init_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, d_query)
#         )
#         self.key_proj = nn.Linear(d_query, d_query)
#         self.value_proj = nn.Linear(d_query, d_query)
        
#         # 多头注意力机制
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=d_query, num_heads=num_heads)
        
#         # 输出的线性变换层
#         self.output_proj = nn.Linear(d_query, d_query)
    
#     def forward(self, queries, features):
#         """
#         :param queries: (batchsize, 20, q_init_dim)  Queries, 20个query
#         :param features: (batchsize, d_query)  特征
#         :return: (batchsize, d_query) 融合后的特征
#         """

#         # 将features扩展为 (batchsize, 1, d_query) 作为自注意力的值 (Value)
#         features = features.unsqueeze(1)  # 变成 (batchsize, 1, d_query)
        
#         # 将queries进行线性映射
#         Q = self.query_proj(queries)  # (batchsize, 20, d_query)
#         K = self.key_proj(features)    # (batchsize, 1, d_query)
#         V = self.value_proj(features)  # (batchsize, 1, d_query)
        
#         # 计算多头注意力
#         Q = Q.transpose(0, 1)  # 转换为 (20, batchsize, d_query)
#         K = K.transpose(0, 1)  # 转换为 (1, batchsize, d_query)
#         V = V.transpose(0, 1)  # 转换为 (1, batchsize, d_query)
#         attn_output, _ = self.multihead_attn(Q, K, V)  # (20, batchsize, d_query)
        
#         # 转换回 (batchsize, 20, d_query)
#         attn_output = attn_output.transpose(0, 1)
        
#         # 输出层进行线性映射
#         output = self.output_proj(attn_output.mean(dim=1))  # (batchsize, d_query)
        
        return output

class AngularQEncoder(nn.Module):
    def __init__(self, q_init_dim=2, d_query=512, num_heads=8):
        super(AngularQEncoder, self).__init__()
        self.q_init_dim = q_init_dim
        self.d_query = d_query
        self.num_heads = num_heads
        
        # 线性变换层
        self.query_proj = nn.Sequential(
            nn.Linear(q_init_dim, 256),
            nn.ReLU(),
            nn.Linear(256, d_query)
        )
        self.key_proj = nn.Linear(d_query, d_query)
        self.value_proj = nn.Linear(d_query, d_query)
        
        # 输出的线性变换层
        self.output_proj = nn.Linear(d_query, d_query)
    
    def forward(self, queries, features):
        """
        :param queries: (batchsize, 20, q_init_dim)  Queries, 20个query
        :param features: (batchsize, d_query)  特征
        :return: (batchsize, d_query) 融合后的特征
        """

        # 将features扩展为 (batchsize, 1, d_query) 作为自注意力的值 (Value)
        features = features.unsqueeze(1)  # 变成 (batchsize, 1, d_query)
        
        # 将queries进行线性映射
        Q = self.query_proj(queries)  # (batchsize, 20, d_query)
        K = self.key_proj(features)    # (batchsize, 1, d_query)
        V = self.value_proj(features)  # (batchsize, 1, d_query)
        
        # 计算自注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_query**0.5  # (batchsize, 20, 1)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batchsize, 20, 1)
        
        # 计算加权的值（Value）
        attended_features = torch.matmul(attention_weights, V)  # (batchsize, 20, d_query)
        
        # 融合后的特征（这里直接使用attended_features）
        # fused_features = attended_features.squeeze(1)  # (batchsize, d_query)
        
        # 输出层进行线性映射
        output = self.output_proj(attended_features.mean(dim=1))  # (batchsize, d_query)
        
        return output

import time


if __name__ == "__main__":
    model = FCRN_encoder(layers=50, decoder='fasterupproj', output_size=(128, 128))
    model = model.cuda(2)
    model.eval()
    image = torch.randn(16, 2, 257, 166)
    image = image.cuda(2)

    gpu_time = time.time()
    with torch.no_grad():
        output = model(image)
    gpu_time = time.time() - gpu_time
    print('gpu_time = ', gpu_time)
    print(output.size())
    print(output[0].size())