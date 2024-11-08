# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder
from ..initializer import xavier_uniform_, linear_init_
from ..layers import MultiHeadAttention
from paddle import ParamAttr
from paddle.regularizer import L2Decay

__all__ = ['HybridEncoder']

@register
class TransformerLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.cross_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src,memory, src_mask=None, pos_embed=None):
        residual =src
        if self.normalize_before:
            src = self.norm1(src)
        q  = self.with_pos_embed(src, pos_embed)
        k  = self.with_pos_embed(memory, pos_embed)

        src = self.cross_attn(q, k, value=memory, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

