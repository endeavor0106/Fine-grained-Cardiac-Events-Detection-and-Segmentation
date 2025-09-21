from typing import Optional, Callable, List, Any

import numpy as np
import torch
import torchvision.ops
from einops import rearrange
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    # [64, 161, 128] [64, 84, 256]
    x = x.view(B, L // window_size, window_size, C)
    # [64, 23, 7, 128] [64, 12, 7, 256]
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    # [1472, 7, 128] [768, 7, 256]
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        L (int): Sequence length

    Returns:
        x: (B, L, C)
    """
    # [1472, 7, 128]
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    # [64, 23, 7, 128]
    x = x.permute(0, 1, 2, 3).contiguous().view(B, L, -1)
    # [64, 161, 128]
    return x


class WindowAttention1D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wl
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))  # 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_l = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(
            [coords_l], indexing='ij'))  # 1, Wl
        coords_flatten = torch.flatten(coords, 1)  # 1, Wl
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 1, Wl, Wl
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wl, Wl, 2
        relative_coords[:, :, 0] += self.window_size - \
            1  # shift to start from 0
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wl, Wl
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wl, Wl) or None
        """
        B_, N, C = x.shape
        # [1472, 7, 128]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # [3, 1472, 4, 7, 32]

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # [1472, 4, 7, 7]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # Wl,Wl,nH
        # [7, 7, 4]

        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wl, Wl
        # [4, 7, 7]

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # attn: [1472, 4, 7, 7]

        attn = self.attn_drop(attn)
        # [1472, 4, 7, 7]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # [1472, 7, 128]
        x = self.proj(x)
        # [1472, 7, 128]
        x = self.proj_drop(x)
        # [1472, 7, 128]
        return x


class SwinTransformerBlock1D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention1D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, L, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        # [64, 160, 128]
        x = self.norm1(x)
        # [64, 160, 128]

        # pad feature maps to multiples of window size
        pad_l = 0
        pad_r = (window_size - L % window_size) % window_size
        x = F.pad(x, (0, 0, pad_l, pad_r))
        # [64, 161, 128]
        _, Lp, _ = x.shape
        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=-shift_size, dims=(1))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # [64, 161, 128]

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wl, C
        # [1472, 7, 128]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wl, C
        # [1472, 7, 128]

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size, C))
        # [1472, 7, 128]
        shifted_x = window_reverse(
            attn_windows, window_size, Lp)  # B D' H' W' C
        # [64, 161, 128]

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1))
        else:
            x = shifted_x
        # [64, 161, 128]

        if pad_r > 0:
            x = x[:, :L, :].contiguous()
        # [64, 160, 128]
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, L, C).
        """
        B, L, C = x.shape

        # padding
        pad_input = (L % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, L % 2))

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


def compute_mask(L, window_size, shift_size, device):
    Lp = int(np.ceil(L / window_size)) * window_size
    img_mask = torch.zeros((1, Lp, 1), device=device)  # 1 Lp 1
    pad_size = int(Lp - L)
    if (pad_size == 0) or (pad_size + shift_size == window_size):
        segs = (slice(-window_size), slice(-window_size, -
                shift_size), slice(-shift_size, None))
    elif pad_size + shift_size > window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-seg1), slice(-seg1, -window_size),
                slice(-window_size, -shift_size), slice(-shift_size, None))
    elif pad_size + shift_size < window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-window_size), slice(-window_size, -seg1),
                slice(-seg1, -shift_size), slice(-shift_size, None))
    cnt = 0
    for d in segs:
        img_mask[:, d, :] = cnt
        cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws, 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock1D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, L = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        x = rearrange(x, 'b c l -> b l c')
        # Lp = int(np.ceil(L / window_size)) * window_size
        attn_mask = compute_mask(L, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, L, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b l c -> b c l')
        return x


class PatchEmbed1D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=32, embed_dim=128, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, L = x.size()
        # [64, 9, 640]
        if L % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - L % self.patch_size))
        # [64, 9, 640]
        x = self.proj(x)  # B C Wl
        # [64, 128, 160]
        if self.norm is not None:
            Wl = x.size(2)
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wl)
        # [64, 128, 160]
        return x


class SwinTransformerFPN(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 num_classes=9,
                 patch_size=4,
                 in_chans=32,
                 embed_dim=128,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=63,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed1D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers-1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self.time_length = 160 // 2**(self.num_layers-1)

        self.MultiscaleBoundaryEnhancementModule = MultiscaleBoundaryEnhancementModule(input_channels=self.num_features, num_classes=4, hidden_dim=self.num_features)

        self.classifierLinear = nn.Sequential(
            # nn.Linear(1024*20, 4*100)
            nn.Linear(self.num_features * self.time_length, 4 * 100)
        )

        self.Decoder = PQRST_SegNet(4)

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        batchsize, num_channels, length = x.size()
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = self.MultiscaleBoundaryEnhancementModule(x)

        x = rearrange(x, 'n c l -> n l c')
        x = self.norm(x)
        x = rearrange(x, 'n l c -> n c l')

        # use CNN decoder
        # x = self.Decoder(x)

        # use linear
        x = x.reshape(batchsize, -1)
        x = self.classifierLinear(x)
        x = x.reshape(batchsize, 4, 100)

        x = F.softmax(x, dim=1)

        return x


class CBR_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=9, stride=1, padding=4):
        super().__init__()
        self.seq_list = [
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()]

        self.seq = nn.Sequential(*self.seq_list)

    def forward(self, x):
        return self.seq(x)


class PQRST_SegNet(nn.Module):
    def __init__(self, class_n):
        super().__init__()
        # self.conv1 = nn.Conv1d(4 * point_nums, 4, kernel_size=5, stride=1, padding=2)
        self.CNN2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=46, stride=6, padding=0)

        ### ------- decoder -----------
        self.upsample_5 = nn.ConvTranspose1d(512, 512, kernel_size=8, stride=2, padding=3)
        self.dec5_1 = CBR_1D(512, 256)
        self.dec5_2 = CBR_1D(256, 256)

        self.upsample_4 = nn.ConvTranspose1d(256, 256, kernel_size=8, stride=2, padding=3)
        self.dec4_1 = CBR_1D(256, 128)
        self.dec4_2 = CBR_1D(128, 128)

        self.upsample_3 = nn.ConvTranspose1d(128, 128, kernel_size=8, stride=2, padding=3)
        self.dec3_1 = CBR_1D(128, 64)
        self.dec3_2 = CBR_1D(64, 64)

        self.upsample_2 = nn.ConvTranspose1d(64, 64, kernel_size=8, stride=2, padding=3)
        self.dec2_1 = CBR_1D(64, 32)
        self.dec2_2 = CBR_1D(32, 32)

        self.upsample_1 = nn.ConvTranspose1d(32, 32, kernel_size=8, stride=2, padding=3)
        self.dec1_1 = CBR_1D(32, 16)
        self.dec1_2 = CBR_1D(16, 16)
        self.dec1_3 = CBR_1D(16, 8)
        self.dec1_4 = CBR_1D(8, 4)


    def forward(self, x):
        # [64, 512, 40]
        dec5 = self.upsample_5(x)
        dec5 = self.dec5_1(dec5)

        dec4 = self.upsample_4(dec5)
        dec4 = self.dec4_1(dec4)

        dec3 = self.upsample_3(dec4)
        dec3 = self.dec3_1(dec3)

        dec2 = self.upsample_2(dec3)
        dec2 = self.dec2_1(dec2)

        dec1 = self.dec1_1(dec2)
        dec1 = self.dec1_3(dec1)

        dec0 = self.dec1_4(dec1)

        # [64, 4, 640]
        out = self.CNN2(dec0)

        return out


class MultiscaleBoundaryEnhancementModule(nn.Module):
    def __init__(self, input_channels=512, num_classes=4, hidden_dim=512):
        super(MultiscaleBoundaryEnhancementModule, self).__init__()
        self.multi_scale_module = MultiScaleFeatureExtraction(input_channels, hidden_dim)
        self.context_encoder = ContextualEncoder(hidden_dim)
        self.boundary_detection = BoundaryAwareModule2(hidden_dim)
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.multi_scale_module(x)
        attended_features = features
        contextual_features = self.context_encoder(attended_features)
        boundary_features = self.boundary_detection(contextual_features)
        logits = boundary_features
        return logits


class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(MultiScaleFeatureExtraction, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(out)

class ContextualEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ContextualEncoder, self).__init__()

        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        self.local_context = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_permuted)
        lstm_out = lstm_out.permute(0, 2, 1)

        local_out = self.local_context(x)

        return lstm_out + local_out


class BoundaryAwareModule2(nn.Module):
    def __init__(self, hidden_dim):
        super(BoundaryAwareModule2, self).__init__()

        self.boundary_detector = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.boundary_enhancement = nn.Sequential(
            nn.Conv1d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv1d(1, 1, 11, padding=5),
            nn.Sigmoid()
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )

        self.enhance_transform = nn.Sequential(
            nn.Conv1d(hidden_dim + 1, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.Dropout(0.1)
        )

        self.gate = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        boundary_map = self.boundary_detector(x)

        spatial_weights = self.spatial_att(boundary_map)
        x_spatial = x * spatial_weights

        channel_weights = self.channel_att(x_spatial)
        x_channel = x_spatial * channel_weights

        enhanced = self.enhance_transform(torch.cat([x_channel, boundary_map], dim=1))

        gate = self.gate(enhanced)
        return x * (1 - gate) + enhanced * gate


if __name__ == "__main__":
    net = SwinTransformerFPN(
        num_classes=4,
        in_chans=9,
        embed_dim=128,
        depths=[2, 2, 6],  # [2, 2, 6, 2]
        num_heads=[4, 8, 16],  # [4, 8, 16, 32]
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False)
    x = torch.rand(64, 9, 640)
    print(net(x).shape)
