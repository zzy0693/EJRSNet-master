import paddle
from paddle import nn
from functools import partial

class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None,act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class DilateAttention(nn.Layer):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_sizes=kernel_size, strides=1, paddings = dilation * (kernel_size-1) // 2, dilations=dilation)
        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self,q, k, v):
        #B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d//self.head_dim,self.head_dim, 1 ,H*W]).transpose((0, 1, 4, 3, 2))  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).transpose((0, 1, 4, 2, 3))  #B, h, N, d, k*k
        attn = (q @ k) * self.scale  # B, h, N, 1, k*k
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//64, 64, self.kernel_size*self.kernel_size, H*W]).transpose((0, 1, 4, 3, 2))  # B, h, N, k*k, d
        x = (attn @ v).transpose((0, 2, 1, 3, 4)).reshape((B, H, W, d))
        return x
class MultiDilatelocalAttention(nn.Layer):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.,proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2D(dim, dim * 3, 1, bias_attr=qkv_bias)
        self.dilate_attention = nn.LayerList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.transpose((0, 3, 1, 2))# B, C, H, W
        qkv = self.qkv(x).reshape((B, 3, self.num_dilation, C//self.num_dilation, H, W)).transpose((2, 1, 0, 3, 4, 5))
        x = x.reshape((B, self.num_dilation, C//self.num_dilation, H, W)).transpose((1, 0, 3, 4, 2))
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2]) # B, H, W,C//num_dilation

        x = x.transpose((1, 2, 3, 0, 4)).reshape((B, H, W, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DilateBlock(nn.Layer):
    "Implementation of Dilate-attention block"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[2, 4, 6, 8]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.transpose((0, 2, 3, 1))  # B H W C
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose((0, 3, 1, 2))
        # B, C, H, W
        return x


    def forward(self, x):

        x = x + self.pos_embed(x)

        x = x.transpose((0, 2, 3, 1))   # B H W C
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose((0, 3, 1, 2))

        return x
class PatchEmbed(nn.Layer):
    """Image to Patch Embedding.
    """
    def __init__(self, img_size=224, in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        # Conv stem
        self.proj = nn.Sequential(
            nn.Conv2D(in_chans, hidden_dim, kernel_size=3, stride=1,
                        padding=1, bias_attr=False),  # 224x224
            nn.BatchNorm2D(hidden_dim),
            nn.GELU( ),
            nn.Conv2D(hidden_dim, int(hidden_dim*2), kernel_size=3, stride=2,
                        padding=1, bias_attr=False),  # 112x112
            nn.BatchNorm2D(int(hidden_dim*2)),
            nn.GELU( ),
            nn.Conv2D(int(hidden_dim*2), int(hidden_dim*4), kernel_size=3, stride=1,
                        padding=1, bias_attr=False),  # 112x112
            nn.BatchNorm2D(int(hidden_dim*4)),
            nn.GELU( ),
            nn.Conv2D(int(hidden_dim*4), embed_dim, kernel_size=3, stride=2,
                        padding=1, bias_attr=False),  # 56x56
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B, C, H, W
        return x
class PatchMerging(nn.Layer):
    """ Patch Merging Layer.
    """
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2D):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            norm_layer(out_channels),
        )

    def forward(self, x):
        #x: B, C, H ,W
        x = self.proj(x)
        return x
class Dilatestage(nn.Layer):
    """ A basic Dilate Transformer layer for one stage.
    """
    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, downsample=True):

        super().__init__()
        # build blocks
        self.blocks = nn.LayerList([
            DilateBlock(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2)) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x



class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)