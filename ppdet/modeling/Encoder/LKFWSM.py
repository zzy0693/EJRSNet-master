import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F


class LKFWSM(nn.Layer):
    def __init__(self, in_channel, ratio=4):
        super(LKFWSM, self).__init__()

        self.ratio = ratio
        self.LSKblock = LKFblock(in_channel // self.ratio)

    @staticmethod
    def channel_shuffle(x, ratio):
        b, c, h, w = x.shape

        x = x.reshape((b, ratio, -1, h, w))
        x = x.transpose([0, 2, 1, 3, 4])

        # flatten
        x = x.reshape((b, -1, h, w))
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape((b * self.ratio, -1, h, w))
        out = self.LSKblock(x)

        out = out.reshape((b, -1, h, w))
        out = self.channel_shuffle(out, 2)
        return out


class LKFblock(nn.Layer):
    def __init__(self, dim, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim // 2, 1)
        self.conv2 = nn.Conv2D(dim, dim // 2, 1)
        self.conv = nn.Conv2D(dim, dim, 1)
        self.conv11 = nn.Conv2D(dim, dim // 2, 1)

        self.conv_1 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                                stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                                stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                                stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                                stride=stride, groups=conv_groups[3])

        self.conv_11 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                                 stride=stride, groups=conv_groups[0])
        self.conv_22 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                                 stride=stride, groups=conv_groups[1])
        self.conv_33 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                                 stride=stride, groups=conv_groups[2])
        self.conv_44 = nn.Conv2D(dim // 2, dim // 8, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                                 stride=stride, groups=conv_groups[3])
        self.FWE = FWEBlock(dim // 8)
        self.split_channel = dim // 8
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # b, c, h, w = x.shape
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = paddle.concat([attn1, attn2], axis=1)
        attn = self.conv11(attn)

        batch_size = attn1.shape[0]
        # Group conv
        x1 = self.conv_1(attn1)
        x2 = self.conv_2(attn1)
        x3 = self.conv_3(attn1)
        x4 = self.conv_4(attn1)

        feats1 = paddle.concat((x1, x2, x3, x4), axis=1)
        feats1 = feats1.reshape([batch_size, 4, self.split_channel, feats1.shape[2], feats1.shape[3]])

        x1 = self.conv_1(attn2)
        x2 = self.conv_2(attn2)
        x3 = self.conv_3(attn2)
        x4 = self.conv_4(attn2)

        feats2 = paddle.concat((x1, x2, x3, x4), axis=1)
        feats2 = feats2.reshape([batch_size, 4, self.split_channel, feats2.shape[2], feats2.shape[3]])

        # stage 2
        # Group conv
        x11 = self.conv_11(attn)
        x22 = self.conv_22(attn)
        x33 = self.conv_33(attn)
        x44 = self.conv_44(attn)

        x1_se = self.FWE(x11)
        x2_se = self.FWE(x22)
        x3_se = self.FWE(x33)
        x4_se = self.FWE(x44)

        x_se = paddle.concat((x1_se, x2_se, x3_se, x4_se), axis=1)
        attention_vectors = x_se.reshape([batch_size, 4, self.split_channel, 1, 1])
        attention_vectors = self.softmax(attention_vectors)  # stage 3

        # stage 4
        feats_weight1 = feats1 * attention_vectors
        feats_weight2 = feats2 * attention_vectors
        for i in range(4):
            x_se_weight_fp1 = feats_weight1[:, i, :, :]
            x_se_weight_fp2 = feats_weight2[:, i, :, :]
            if i == 0:
                attn1 = x_se_weight_fp1
                attn2 = x_se_weight_fp2
            else:
                attn1 = paddle.concat((x_se_weight_fp1, attn1), axis=1)
                attn2 = paddle.concat((x_se_weight_fp2, attn2), axis=1)

        attn = paddle.concat([attn1, attn2], axis=1)
        attn = self.conv(attn) * x

        return attn


class FWEBlock(nn.Layer):
    def __init__(self, in_channels, scale=16):
        super(FWEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2D(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(axis=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2D(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).reshape([b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
        query = x.reshape([b, c, h * w])
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = paddle.matmul(query, key)
        concate_QK = concate_QK.reshape([b, c, 1, 1])
        value = self.Conv_value(concate_QK)

        return value