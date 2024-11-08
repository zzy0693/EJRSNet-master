
import paddle
import paddle.nn as nn
from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock


class TRM1RepLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(TRM1RepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

        self.conv_squeeze = nn.Conv2D(2, 2, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = paddle.concat([x_1, x_2], axis=1)
        avg_attn = paddle.mean(x, axis=1, keepdim=True)
        max_attn= paddle.max(x, axis=1, keepdim=True)
        agg = paddle.concat([avg_attn, max_attn], axis=1)
        sig = self.conv_squeeze(agg)
        sig =  self.sigmoid(sig)
        attn = x_1 * sig[:, 0, :, :].unsqueeze(1)+ x_2 * sig[:, 1, :, :].unsqueeze(1)
        return self.conv3(attn)