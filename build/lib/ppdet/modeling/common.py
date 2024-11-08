import paddle.nn.functional as F
import paddle
import paddle.nn as nn

class self_SENet(nn.Layer):
    def __init__(self, in_channel, ratio=2):
        super(self_SENet, self).__init__()

        self.ratio = ratio
        self.LSKblock = LSKblock(in_channel // self.ratio)

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

class LSKblock(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim , 1)
        self.conv2 = nn.Conv2D(dim, dim , 1)
        # self.conv_squeeze = nn.Conv2D(2, 2, 7, padding=3)
        self.conv = nn.Conv2D(dim , dim, 1)
        # self.sigmoid = nn.Sigmoid()

        self.GCSEV2=GCSEV2(dim*2)
    def forward(self, x):
        # b, c, h, w = x.shape
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # attn1 = self.SEWeightModule(attn1)
        # attn2 = self.SEWeightModule(attn2)

        attn = paddle.concat([attn1, attn2], axis=1)

        attn3=self.GCSEV2(attn)
        # attn3=self.conv1(attn3)
        attn1=attn1*attn3*attn2
        # attn2=attn2+attn3
        # attn=paddle.concat([attn1,attn2], axis=1)

        # avg_attn = paddle.mean(attn, axis=1, keepdim=True)
        # max_attn= paddle.max(attn, axis=1, keepdim=True)
        # agg = paddle.concat([avg_attn, max_attn], axis=1)
        # sig = self.conv_squeeze(agg)
        # sig =  self.sigmoid(sig)
        # attn = attn1 * sig[:, 0, :, :].unsqueeze(1)+ attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv (attn1)*x

        return  attn

class SEWeightModule(nn.Layer):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = nn.Conv2D(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)+self.max_pool (x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight  = self.sigmoid(out)

        weight=weight*x
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

import paddle
import paddle.nn as nn

class Conv(nn.Layer):
    # 包含BN和ReLU
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,bias_attr=True)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWR(nn.Layer):
    def __init__(self, c) -> None:
        super().__init__()

        self.conv_3x3 = Conv(c, c, 3, padding=1)

        self.conv_3x3_d1 = Conv(c, c, 3, padding=1, dilation=1)
        self.conv_3x3_d3 = Conv(c, c, 3, padding=3, dilation=3)
        self.conv_3x3_d5 = Conv(c, c, 3, padding=5, dilation=5)

        self.conv_1x1 = Conv(c * 3, c, 1)

    def forward(self, x):
        x_ = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(x_)
        x2 = self.conv_3x3_d3(x_)
        x3 = self.conv_3x3_d5(x_)

        x_out = paddle.concat([x1, x2, x3], axis=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out

class DWRSeg_Conv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super(DWRSeg_Conv, self).__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size=1)

        self.dcnv3 = DWR(out_channels)

        self.bn = nn.BatchNorm(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)

        x = self.dcnv3(x)

        x = self.gelu(self.bn(x))
        return x


class SPDConv(nn.Layer):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.Silu()  # default activation

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super(SPDConv, self).__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2D(c1, c2, k, s, groups=g, dilation=d, bias_attr=False)
        self.bn = nn.BatchNorm(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Layer) else nn.Identity()

    def forward(self, x):
        x = paddle.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = paddle.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))



class ASPP(nn.Layer):
    def __init__(self, in_channel, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2D((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2D(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2D(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2D(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2D(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2D(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2D(depth * 5, depth, 1, 1)

    def forward(self, x,x2):
        size = x.shape[2:]

        image_features = self.mean(x2)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features,size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(paddle.concat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], axis=1))
        return net

class SEV2(nn.Layer):
    def __init__(self, in_channels, scale=16,numbers=4):
        super(SEV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale
        self.numbers = numbers
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, kernel_size=1, padding=0),
            nn.ReLU(),

        )

        self.fc2 = nn.Sequential(
            nn.Conv2D(self.out_channels*self.numbers, self.in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)+self.max_pool(x)
        value1= self.fc1(y)
        for i in range(1, self.numbers):
            y = self.avg_pool(x)
            value2 = self.fc1(y)
            value1 = paddle.concat([value1, value2], axis=1)
        out=self.fc2(value1)
        # out = x * value1
        return out

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
            nn.Conv2D(self.out_channels*self.numbers, self.in_channels//2, 1),
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
        out=self.Conv_value2(value1)
        # out = x + value1
        return out


class SoftPool2D(nn.Layer):
    def __init__(self, kernel_size, stride):
        super(SoftPool2D,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None):
        kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = (stride, stride)
        _, c, h, w = x.shape
        e_x = paddle.sum(paddle.exp(x),axis=1,keepdim=True)
        return F.avg_pool2d(x * e_x, kernel_size, stride=stride) * (sum(kernel_size))/(F.avg_pool2d(e_x, kernel_size, stride=stride) * (sum(kernel_size)))


class AttentionModule(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2D(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2D(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2D(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2D(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2D(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2D(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

