import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 多头注意力机制的头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = head_dim ** -0.5  # 归一化参数

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 产生qkv
        self.attn_drop = nn.Dropout(attn_drop)  # attention_score的dropout
        self.proj = nn.Linear(dim, dim)  # 多头注意力合并之后的语义空间转化
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
