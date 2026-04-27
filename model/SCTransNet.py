# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : SCTransNet.py
# @Software: PyCharm
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch import Tensor
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile



def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    # def forward(self, x, h, w):
    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# spatial-embedded Single-head Channel-cross Attention (SSCA)
class Attention_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_cross, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.text_size = config.text_size
        self.hidden_size = config.hidden_size
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mheadk1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)

        self.mheadv1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)


        self.k1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.k2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.k3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.k4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        self.v1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.v2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.v3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.v4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        self.q = nn.Linear(self.text_size, self.hidden_size)
        
       
        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)


       

    def forward(self, emb1, emb2, emb3, emb4, text_emb):
        b, c, h, w = emb1.shape
        q=self.q(text_emb)

        k1 = self.k1(self.mheadk1(emb1))
        k2 = self.k2(self.mheadk2(emb2))
        k3 = self.k3(self.mheadk3(emb3))
        k4 = self.k4(self.mheadk4(emb4))

        v1 = self.v1(self.mheadv1(emb1))
        v2 = self.v2(self.mheadv2(emb2))
        v3 = self.v3(self.mheadv3(emb3))
        v4 = self.v4(self.mheadv4(emb4))
        
        # k, v = kv.chunk(2, dim=1)

        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k3 = rearrange(k3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k4 = rearrange(k4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v3 = rearrange(v3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v4 = rearrange(v4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        k3 = torch.nn.functional.normalize(k3, dim=-1)
        k4 = torch.nn.functional.normalize(k4, dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1)
        _, _, c1, _ = k1.shape
        _, _, c2, _ = k2.shape
        _, _, c3, _ = k3.shape
        _, _, c4, _ = k4.shape
        _, _, c, _ = q.shape

        attn1 = (q @ k1.transpose(-2, -1)) / math.sqrt(c1)
        attn2 = (q @ k2.transpose(-2, -1)) / math.sqrt(c2)
        attn3 = (q @ k3.transpose(-2, -1)) / math.sqrt(c3)
        attn4 = (q @ k4.transpose(-2, -1)) / math.sqrt(c4)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        out1 = (attention_probs1 @ v1).mean(dim=1)
        out2 = (attention_probs2 @ v2).mean(dim=1)
        out3 = (attention_probs3 @ v3).mean(dim=1)
        out4 = (attention_probs4 @ v4).mean(dim=1)

        

        out1 = rearrange(out1, 'b  c (h w) -> b c h w', h=h, w=w)
        out2 = rearrange(out2, 'b  c (h w) -> b c h w', h=h, w=w)
        out3 = rearrange(out3, 'b  c (h w) -> b c h w', h=h, w=w)
        out4 = rearrange(out4, 'b  c (h w) -> b c h w', h=h, w=w)

        out1 = self.project_out1(out1)
        out2 = self.project_out2(out2)
        out3 = self.project_out3(out3)
        out4 = self.project_out4(out4)
        weights = None
        

        return out1, out2, out3, out4, weights

class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)


        # ****************** useless ***************************************
        # self.q1_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q1_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q1_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q1_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        # self.q2_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q2_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q2_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q2_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        # self.q3_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q3_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q3_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q3_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        # self.q4_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q4_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q4_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.q4_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))
        # k, v = kv.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q4 = rearrange(q4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)
        q4 = torch.nn.functional.normalize(q4, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c2, _ = q2.shape
        _, _, c3, _ = q3.shape
        _, _, c4, _ = q4.shape
        _, _, c, _ = k.shape

        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn4 = (q4 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        out1 = (attention_probs1 @ v).mean(dim=1)
        out2 = (attention_probs2 @ v).mean(dim=1)
        out3 = (attention_probs3 @ v).mean(dim=1)
        out4 = (attention_probs4 @ v).mean(dim=1)

        

        out1 = rearrange(out1, 'b  c (h w) -> b c h w', h=h, w=w)
        out2 = rearrange(out2, 'b  c (h w) -> b c h w', h=h, w=w)
        out3 = rearrange(out3, 'b  c (h w) -> b c h w', h=h, w=w)
        out4 = rearrange(out4, 'b  c (h w) -> b c h w', h=h, w=w)

        out1 = self.project_out1(out1)
        out2 = self.project_out2(out2)
        out3 = self.project_out3(out3)
        out4 = self.project_out4(out4)
        weights = None
        

        return out1, out2, out3, out4, weights

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        x_3,x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)
        return x


#  Spatial-channel Cross Transformer Block (SCTB)
class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)


    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 196 960
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights
    
class Dblock_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Dblock_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.text_size, LayerNorm_type='WithBias')

        self.channel_attn = Attention_cross(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)


    def forward(self, emb1, emb2, emb3, emb4, emb_text):
        

        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_text = self.attn_norm(emb_text)  # 1 196 960
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_text)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights
    
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim_q: int,
        embedding_dim_kv: int,
        num_heads: int,
        downsample_q_rate: int = 1,
        
    ) -> None:
        super().__init__()
        self.embedding_dim_q = embedding_dim_q
        self.internal_dim = embedding_dim_q // downsample_q_rate
        self.embedding_dim_kv = embedding_dim_kv
     

        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        

        self.q_proj = nn.Linear(self.embedding_dim_q, self.internal_dim)
        self.k_proj = nn.Linear(self.embedding_dim_kv, self.internal_dim)
        self.v_proj = nn.Linear(self.embedding_dim_kv, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.embedding_dim_kv)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor,is_return_attn=False) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn_norm = torch.softmax(attn, dim=-1)

        # Get output
        out = attn_norm @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        if is_return_attn:
            return out, attn, attn_norm
        else:
            return out
    
class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead,d_kv, downsample_q_rate=1,dim_feedforward=2048, dropout=0.1,
                 activation="relu", SAtt=True, is_feedforward=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation,
                                                      batch_first=True)
        self.SAtt = SAtt
        self.is_feedforward = is_feedforward
        self.multihead_attn = Attention(embedding_dim_q=d_model,
                                        embedding_dim_kv=d_kv,
                                        num_heads=nhead,
                                        downsample_q_rate=downsample_q_rate)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,is_return_attn=False):
        if self.SAtt:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
        if is_return_attn:
            tgt2, attn, attn_norm = self.multihead_attn(tgt, memory, memory,is_return_attn=is_return_attn)
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory,is_return_attn=is_return_attn)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.is_feedforward:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        if is_return_attn:
            return tgt, attn, attn_norm
        else:
            return tgt


class Encoder(nn.Module):

    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights

class Decoder(nn.Module):
    
    def __init__(self, config, vis, channel_num,is_spitial_transformer_decoder=False):
        super(Decoder, self).__init__()
        self.is_spitial_transformer_decoder=is_spitial_transformer_decoder
        self.is_satt=config.transformer["SATT"]
        self.is_feedforward=config.transformer["IS_FEEDFORWARD"]
        self.vis = vis
        self.layer = nn.ModuleList()
        self.decoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.decoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.decoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.decoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_delayers"]):
            layer = Dblock_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))
        if self.is_spitial_transformer_decoder:
          
            self.decoder1=cross_filter(config,512,32,16,32,1,32,0,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=config.IS_T2V)
            self.decoder2=cross_filter(config,512,64,8,64,1,64,0,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=config.IS_T2V)
            self.decoder3=cross_filter(config,512,128,4,128,1,128,0,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=config.IS_T2V)
            self.decoder4=cross_filter(config,512,256,2,256,1,256,0,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=config.IS_T2V)
                                      
                                       

    def forward(self, emb1, emb2, emb3, emb4,emb_text):
        attn_weights = []
        if self.is_spitial_transformer_decoder:
            emb1=self.decoder1(emb1,emb_text)
            emb2=self.decoder2(emb2,emb_text)
            emb3=self.decoder3(emb3,emb_text)
            emb4=self.decoder4(emb4,emb_text)
         
        else:
            for layer_block in self.layer:
                emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
                if self.vis:
                    attn_weights.append(weights)
            emb1 = self.decoder_norm1(emb1) if emb1 is not None else None
            emb2 = self.decoder_norm2(emb2) if emb2 is not None else None
            emb3 = self.decoder_norm3(emb3) if emb3 is not None else None
            emb4 = self.decoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights

class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)
        self.decoder = Decoder(config, vis, channel_num,is_spitial_transformer_decoder=config.is_spitial_transformer_decoder)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4, text_embd=None):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        emb1, emb2, emb3, emb4, attn_weights = self.encoder(emb1, emb2, emb3, emb4)  # (B, n_patch, hidden)
        if text_embd is not None:
            emb1, emb2, emb3, emb4, attn_weights = self.decoder(emb1, emb2, emb3, emb4, text_embd)

        emb1 = self.reconstruct_1(emb1) if en1 is not None else None
        emb2 = self.reconstruct_2(emb2) if en2 is not None else None
        emb3 = self.reconstruct_3(emb3) if en3 is not None else None
        emb4 = self.reconstruct_4(emb4) if en4 is not None else None

        # x1 = x1 + en1 if en1 is not None else None
        # x2 = x2 + en2 if en2 is not None else None
        # x3 = x3 + en3 if en3 is not None else None
        # x4 = x4 + en4 if en4 is not None else None

        return emb1, emb2, emb3, emb4, attn_weights


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
    # def text_forward(self, x):
    #     '''need coding!!!'''
    #     up = self.up(x)
    #     return up
    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        relu_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.relu_output = relu_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        elif self.relu_output:
            x = F.relu(x)
        return x
class cross_filter(nn.Module):
    def __init__(self,config,text_size,visual_size,downsample_rate,model_size,nhead,feedforward_size,dropout,
                 SAtt,feedforward,filter_type='sig',is_t2v=True):
        super().__init__()

        self.filterlayer=TransformerDecoderLayer(d_model=text_size, nhead=nhead,d_kv=visual_size,downsample_q_rate=downsample_rate,
                                                 dim_feedforward=feedforward_size,dropout=dropout, SAtt=SAtt,is_feedforward=feedforward)
        self.is_t2v=is_t2v
        if self.is_t2v:
            self.text2visual=nn.Linear(text_size, visual_size)
            self.filterlayer=TransformerDecoderLayer(d_model=model_size, nhead=nhead,d_kv=visual_size,downsample_q_rate=1,
                                                 dim_feedforward=feedforward_size,dropout=dropout, SAtt=SAtt,is_feedforward=feedforward)
        
        self.filter_type=filter_type
        self.isfilter=config.IS_FILTER
    def forward(self,visual_embd,text_embd,is_return_attn=False):
        b,c,h,w=visual_embd.size()
        if self.is_t2v:
            text_embd2=self.text2visual(text_embd) #b 1 c
        else:
            text_embd2=text_embd

        visual_embd=torch.permute(visual_embd,(0,2,3,1)).reshape(b,-1,c)  # b hw c
        if self.isfilter:
            if is_return_attn:
                out,attn,attn_norm=self.filterlayer(text_embd2,visual_embd,is_return_attn=is_return_attn) #b 1 c
            else:
                out=self.filterlayer(text_embd2,visual_embd,is_return_attn=is_return_attn) #b 1 c
        else:
            out=text_embd2
        if self.filter_type == "mul":
            visual_embd=out @ visual_embd.transpose(-2,-1)  # b 1 hw
            visual_embd=torch.permute(visual_embd,(0,2,1)).reshape(b,1,h,w)
            
        elif self.filter_type == "sig":
            out=torch.sigmoid(visual_embd @ out.permute(0,2,1)) 
            visual_embd=out * visual_embd
            visual_embd=torch.permute(visual_embd,(0,2,1)).reshape(b,c,h,w)
            
        elif self.filter_type == "product":
            visual_embd=out*visual_embd
            visual_embd=torch.permute(visual_embd,(0,2,1)).reshape(b,c,h,w)
        if is_return_attn:
            return visual_embd, attn, attn_norm
        else:
            return visual_embd

class SCTransNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.config = config
        self.vis = vis
        self.is_prompt_learner=config.IS_PROMPT_LEARNER
        self.deepsuper = deepsuper
        self.is_textinskip=config.IS_TEXTINSKIP
        self.is_satt=config.transformer["SATT"]
        self.is_feedforward=config.transformer["IS_FEEDFORWARD"]
        self.dropout=config.Filter_DROPOUT
        
        self.text_size=config.text_size
        
        print('text in skip-connection:',self.is_textinskip)
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # basic channel 64
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.is_wbfilter=config.IS_WBFILTER
        self.is_cafilter=config.IS_CAFILTER
        self.is_textindskip=config.IS_TEXTINDSKIP
        self.is_textindbase=config.IS_TEXTINDBASE
        print('Using wbfilter:',self.is_wbfilter)
        print('Using cafilter:',self.is_cafilter)
        print('Using textindskip for cafilter:',self.is_textindskip)

       
            
        if self.is_wbfilter:
            self.weight_block=MLP(input_dim=512,hidden_dim=256,output_dim=1,num_layers=3,sigmoid_output=False,relu_output=True)
            self.bias_block=MLP(input_dim=512,hidden_dim=256,output_dim=1,num_layers=3,sigmoid_output=False,relu_output=False)
        elif self.is_cafilter:
            self.cafilter5=cross_filter(config,self.text_size,in_channels*8,2,in_channels*8,1,in_channels*8,self.dropout,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=True)
            
            self.cafilter4=cross_filter(config,self.text_size,in_channels*4,4,in_channels*4,1,in_channels*4,self.dropout,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=True)
            self.cafilter3=cross_filter(config,self.text_size,in_channels*2,8,in_channels*2,1,in_channels*2,self.dropout,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=True)

            self.cafilter2=cross_filter(config,self.text_size,in_channels,16,in_channels,1,in_channels,self.dropout,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=True)
            
            self.cafilter1=cross_filter(config,self.text_size,in_channels,16,in_channels,1,in_channels,self.dropout,SAtt=self.is_satt,feedforward=self.is_feedforward,
                                       filter_type=config.Filter_type,is_t2v=True)
            
           
        
    
        
        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    def init_clip(self,clip):
        self.custom_clip = clip
        if not self.is_prompt_learner:  
            self.emb_text=self.custom_clip()
            self.emb_text=self.emb_text.detach()
            del self.custom_clip
            torch.cuda.empty_cache()
    # def init_embtext(self):
    #     self.emb_text=self.custom_clip()
    #     self.emb_text=self.emb_text.detach()
        
    def forward(self, x, feature=None, is_train=True):
        x1 = self.inc(x)  # 64 224 224
        x2 = self.down_encoder1(self.pool(x1))  # 128 112 112
        x3 = self.down_encoder2(self.pool(x2))  # 256 56  56
        x4 = self.down_encoder3(self.pool(x3))  # 512 28  28
        d5 = self.down_encoder4(self.pool(x4))  # 512 14  14
        #  CCT
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        #  CCT
        if self.is_prompt_learner:
            if feature is not None and self.config.IS_VISUAL:
                text_embd = self.custom_clip(feature,is_return_prompt=False)
            else:
                text_embd = self.custom_clip()
            
        else:
            text_embd = self.emb_text
        if text_embd.size(0) != x1.size(0):
            text_embd = text_embd.expand(x1.size(0), -1, -1)  # expand to batch size
        
        if self.is_textinskip:
            x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4, text_embd)
        else:
            x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4, None)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4
        #  Feature fusion
        if self.is_wbfilter:
            weight=self.weight_block(text_embd).unsqueeze(-1)  # b 1 1 1
            bias=self.bias_block(text_embd).unsqueeze(-1)
            d5=d5*weight+bias   # b 1 1 1
        elif self.is_cafilter:
            if self.is_textindbase:
                d5=self.cafilter5(d5,text_embd)
            if self.is_textindskip:
                x4=self.cafilter5(x4,text_embd)
           
        d4 = self.up_decoder4(d5, x4) 
        if self.is_wbfilter:
            d4=d4*weight+bias  
        elif self.is_cafilter:
            if self.is_textindbase:
                d4=self.cafilter4(d4,text_embd)
            if self.is_textindskip:
                x3=self.cafilter4(x3,text_embd)

        d3 = self.up_decoder3(d4, x3)
        if self.is_wbfilter:
            d3=d3*weight+bias
        elif self.is_cafilter:
            if self.is_textindbase:
                d3=self.cafilter3(d3,text_embd)
            if self.is_textindskip:
                x2=self.cafilter3(x2,text_embd)

        d2 = self.up_decoder2(d3, x2)
        if self.is_wbfilter:
            d2=d2*weight+bias
        elif self.is_cafilter:
            if self.is_textindbase:
                d2=self.cafilter2(d2,text_embd)
            if self.is_textindskip:
                x1=self.cafilter2(x1,text_embd)

        d1=self.up_decoder1(d2, x1)
        if self.is_wbfilter:
            d1=d1*weight+bias
            out = self.outc(d1)
        elif self.is_cafilter:  
            if self.is_textindbase:        
                out=self.cafilter1(d1,text_embd)
            else:
                out=d1
            
            if self.config.Filter_type !='mul':
                out=self.outc(out)
        else:
            out = self.outc(d1)
            
        # deep supervision
        if self.deepsuper and is_train:
            
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            # 原始深监督
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            
            return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
            
        else:
            return torch.sigmoid(out)


if __name__ == '__main__':
    config_vit = get_CTranS_config()
    model = SCTransNet(config_vit, mode='train', deepsuper=True)
    model = model
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
