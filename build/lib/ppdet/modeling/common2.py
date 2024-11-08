import paddle
import paddle.nn as nn
import math
import paddle

from ppdet.modeling.initializer import normal_
import math
from paddle.vision.ops import DeformConv2D
import paddle.nn.functional as F

class GCSEV2(nn.Layer):
    def __init__(self, in_channels, scale=16,numbers=4):
        super(GCSEV2, self).__init__()
        self.in_channels = in_channels
        self.numbers = numbers
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2D(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(axis=1)

        self.Conv_value1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
        )

        self.Conv_value2 = nn.Sequential(
            nn.Conv2D(self.out_channels*self.numbers, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).reshape([b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
        query = x.reshape([b, c, h * w])
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = paddle.matmul(query, key)
        concate_QK = concate_QK.reshape([b, c, 1, 1])
        value1 = self.Conv_value1(concate_QK)
        for i in range(1, self.numbers):
            key = self.SoftMax(self.Conv_key(x).reshape([b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
            query = x.reshape([b, c, h * w])
            # [b, c, h*w] * [b, H*W, 1]
            concate_QK = paddle.matmul(query, key)
            concate_QK = concate_QK.reshape([b, c, 1, 1])
            value2 = self.Conv_value1(concate_QK)
            value1 = paddle.concat([value1, value2], axis=1)
        value1=self.Conv_value2(value1)
        out = x + value1
        return out

class SEV2(nn.Layer):
    def __init__(self, in_channels, scale=16,numbers=4):
        super(SEV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale
        self.numbers = numbers
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Sequential(
                nn.Linear(self.in_channels, self.out_channels),#sq
                nn.ReLU(),
                nn.Linear(self.out_channels, self.in_channels),#ex
                nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.out_channels*self.numbers, self.in_channels),  # ex
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c])
        value1= self.fc1(y).reshape([b, c, 1, 1])
        for i in range(1, self.numbers):
            y = self.avg_pool(x).reshape([b, c])
            value2 = self.fc1(y).reshape([b, c, 1, 1])
            value1 = paddle.concat([value1, value2], axis=1)
        value1=self.fc2(value1)
        out = x * value1
        return out

class DCNv2(paddle.nn.Layer):
        def __init__(self, num_classes=256):
            super(DCNv2, self).__init__()

            self.conv1 = paddle.nn.Conv2D(in_channels=256, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
            # self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

            self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=0)
            # self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

            self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=0)

            self.offsets = paddle.nn.Conv2D(64, 18, kernel_size=3, stride=2, padding=1)
            self.mask = paddle.nn.Conv2D(64, 9, kernel_size=3, stride=2, padding=1)
            self.conv4 = DeformConv2D(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=2, padding=1)

            # self.conv4 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 1)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            # x = self.pool1(x)
            # print(x.shape)
            x = self.conv2(x)
            x = F.relu(x)
            # x = self.pool2(x)
            # print(x.shape)

            x = self.conv3(x)
            x = F.relu(x)
            # print(x.shape)

            offsets = self.offsets(x)
            masks = self.mask(x)
            # print(offsets.shape)
            # print(masks.shape)
            x = self.conv4(x, offsets, masks)
            x = F.relu(x)
            # print(x.shape)

            return x

import paddle.nn as nn
import paddle
from functools import reduce

class GCSKConv(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, M=3, r=16, L=32):
        super(GCSKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.LayerList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2D(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias_attr=False),
                                           nn.BatchNorm(out_channels),
                                           nn.ReLU()))
        self.global_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.global_pool2 = nn.AdaptiveMaxPool2D(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2D(out_channels, d, 1, bias_attr=False),
                                 nn.BatchNorm(d),
                                 nn.ReLU())
        self.fc2 = nn.Conv2D(d, out_channels * self.M, 1, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, input):
        batch_size = input.shape[0]
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y + z, output)
        s = self.global_pool(U)+self.global_pool2(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape([batch_size, self.M, self.out_channels, -1])
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, axis=1))
        a_b = list(map(lambda x: x.reshape([batch_size, self.out_channels, 1, 1]), a_b))
        V = list(map(lambda x, y: x * y, output,a_b))
        V = reduce(lambda x, y: x + y, V)
        return V

# x = paddle.randn([8, 32, 24, 24])
# conv = SKConv(32, 32, 1, 2, 16, 32)
# print(conv(x).shape)import paddle
import paddle.nn as nn
import math
import paddle

from ppdet.modeling.initializer import normal_
import math
from paddle.vision.ops import DeformConv2D
import paddle.nn.functional as F

class GCSEV2(nn.Layer):
    def __init__(self, in_channels, scale=16,numbers=4):
        super(GCSEV2, self).__init__()
        self.in_channels = in_channels
        self.numbers = numbers
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2D(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(axis=1)

        self.Conv_value1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
        )

        self.Conv_value2 = nn.Sequential(
            nn.Conv2D(self.out_channels*self.numbers, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).reshape([b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
        query = x.reshape([b, c, h * w])
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = paddle.matmul(query, key)
        concate_QK = concate_QK.reshape([b, c, 1, 1])
        value1 = self.Conv_value1(concate_QK)
        for i in range(1, self.numbers):
            key = self.SoftMax(self.Conv_key(x).reshape([b, 1, -1]).transpose([0, 2, 1]).reshape([b, -1, 1]))
            query = x.reshape([b, c, h * w])
            # [b, c, h*w] * [b, H*W, 1]
            concate_QK = paddle.matmul(query, key)
            concate_QK = concate_QK.reshape([b, c, 1, 1])
            value2 = self.Conv_value1(concate_QK)
            value1 = paddle.concat([value1, value2], axis=1)
        value1=self.Conv_value2(value1)
        out = x + value1
        return out



class DCNv2(paddle.nn.Layer):
        def __init__(self, num_classes=256):
            super(DCNv2, self).__init__()

            self.conv1 = paddle.nn.Conv2D(in_channels=256, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
            # self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

            self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=0)
            # self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

            self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=0)

            self.offsets = paddle.nn.Conv2D(64, 18, kernel_size=3, stride=2, padding=1)
            self.mask = paddle.nn.Conv2D(64, 9, kernel_size=3, stride=2, padding=1)
            self.conv4 = DeformConv2D(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=2, padding=1)

            # self.conv4 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 1)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            # x = self.pool1(x)
            # print(x.shape)
            x = self.conv2(x)
            x = F.relu(x)
            # x = self.pool2(x)
            # print(x.shape)

            x = self.conv3(x)
            x = F.relu(x)
            # print(x.shape)

            offsets = self.offsets(x)
            masks = self.mask(x)
            # print(offsets.shape)
            # print(masks.shape)
            x = self.conv4(x, offsets, masks)
            x = F.relu(x)
            # print(x.shape)

            return x

import paddle.nn as nn
import paddle
from functools import reduce

class GCSKConv(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, M=4, r=16, L=32):
        super(GCSKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.LayerList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2D(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias_attr=False),
                                           nn.BatchNorm(out_channels),
                                           nn.ReLU()))
        self.global_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.global_pool2 = nn.AdaptiveMaxPool2D(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2D(out_channels, d, 1, bias_attr=False),
                                 nn.BatchNorm(d),
                                 nn.ReLU())
        self.fc2 = nn.Conv2D(d, out_channels * self.M, 1, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)

        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.MLP = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(in_channels, int(in_channels / 4)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 4), in_channels),
            nn.ReLU()
        )
        self.c_bn = nn.BatchNorm2D(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Sequential(
            nn.Conv2D(in_channels, int(in_channels / r), 1),
            nn.Conv2D(int(in_channels / r), int(in_channels / r), 3,4, 4),
            nn.Conv2D(int(in_channels / r), 1, 1)
        )
        self.s_bn = nn.BatchNorm2D(1)
    def Channel_Attention(self, x):
        b, c, h, w = x.shape

        f_a = self.avgpool(x)
        f_mlp = self.MLP(f_a)
        c_a = paddle.reshape(f_mlp, [b, c, 1, 1])
        Mc = self.c_bn(c_a)

        return Mc

    def Spatial_Attention(self, x):
        s_a = self.conv_layer(x)
        Ms = self.s_bn(s_a)

        return Ms

    def BAM_Attention(self, x):
        Mc = self.Channel_Attention(x)
        Ms = self.Spatial_Attention(x)

        M = paddle.add(Mc, Ms)
        M = self.sigmoid(M)

        return M
    def forward(self, input):
        batch_size = input.shape[0]
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        U = self.BAM_Attention(U)
        s = self.global_pool(U)+self.global_pool2(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape([batch_size, self.M, self.out_channels, -1])
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, axis=1))
        a_b = list(map(lambda x: x.reshape([batch_size, self.out_channels, 1, 1]), a_b))
        V = list(map(lambda x, y: x * y, output,a_b ))
        V = list(map(lambda x, y: x + y, V, output))
        V = reduce(lambda x, y: x + y, V)
        return V

# x = paddle.randn([8, 32, 24, 24])
# conv = SKConv(32, 32, 1, 2, 16, 32)
# print(conv(x).shape)

import paddle
import paddle.nn as nn


import paddle
import paddle.nn as nn


class BAM_module(nn.Layer):
    def __init__(self, channel, reduction=16, dilation=4):
        super(BAM_module, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.MLP = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(channel, int(channel / 4)),
            nn.ReLU(),
            nn.Linear(int(channel / 4), channel),
            nn.ReLU()
        )
        self.c_bn = nn.BatchNorm2D(channel)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Sequential(
            nn.Conv2D(channel, int(channel / reduction), 1),
            nn.Conv2D(int(channel / reduction), int(channel / reduction), 3,dilation, dilation),
            nn.Conv2D(int(channel / reduction), 1, 1)
        )
        self.s_bn = nn.BatchNorm2D(1)

    def Channel_Attention(self, x):
        b, c, h, w = x.shape

        f_a = self.avgpool(x)
        f_mlp = self.MLP(f_a)
        c_a = paddle.reshape(f_mlp, [b, c, 1, 1])
        Mc = self.c_bn(c_a)

        return Mc

    def Spatial_Attention(self, x):
        s_a = self.conv_layer(x)
        Ms = self.s_bn(s_a)

        return Ms

    def BAM_Attention(self, x):
        Mc = self.Channel_Attention(x)
        Ms = self.Spatial_Attention(x)

        M = paddle.add(Mc, Ms)
        M = self.sigmoid(M)

        return M

    def forward(self, x):
        M_a = self.BAM_Attention(x)
        F1 = M_a * x
        refined_feature = paddle.add(F1, x)
        return refined_feature

class LSKblock(nn.Layer):
    def __init__(self, dim ):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2D(dim, dim, 5, stride=1, padding=2, groups=dim, dilation=1)
        self.conv_spatial2 = nn.Conv2D(dim, dim, 7, stride=1, padding=3, groups=dim, dilation=1)
        self.conv1 = nn.Conv2D(dim, dim//3, 1)
        self.conv2 = nn.Conv2D(dim, dim//3, 1)
        self.conv3 = nn.Conv2D(dim, dim//3, 1)
        self.SENet=SEWeightModule(dim)
        self.conv_squeeze = nn.Conv2D(2, 2, 7, padding=3)
        self.conv = nn.Conv2D(dim//3, dim, 1)

    def forward(self, x):

        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn3 = self.conv_spatial2(attn1)

        attn1 = self.SENet(attn1 )
        attn2 = self.SENet(attn2 )
        attn3 = self.SENet(attn3 )

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn3 = self.conv3(attn3)

        attn = paddle.concat([attn1, attn2,attn3], axis=1)
        avg_attn = paddle.mean(attn, axis=1, keepdim=True)
        max_attn= paddle.max(attn, axis=1, keepdim=True)
        agg = paddle.concat([avg_attn, max_attn], axis=1)
        sig = self.conv_squeeze(agg)
        attn = attn1 * sig[:,-1,:,:].unsqueeze(1)+attn2 * sig[:, 0, :, :].unsqueeze(1)+attn3 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn




class SEWeightModule(nn.Layer):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        weight = self.SoftMax(weight)  # stage 3
        weight = x * weight
        return weight

class SENet(nn.Layer):
    def __init__(self, in_channel, ratio=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
                nn.Linear(in_channel, in_channel // ratio),#sq
                nn.ReLU(),
                nn.Linear(in_channel // ratio, in_channel),#ex
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c])
        y =  self.fc(y).reshape([b, c, 1, 1])
        return x * y #f归一化