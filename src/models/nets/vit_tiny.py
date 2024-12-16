import torch
import torch.nn as nn
from .patch_embed import PatchEmbed
from .attention import Attention
from .mlp import MLP
from .block import Block

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, num_classes=10, embed_dim=256, depth=2, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x

def vit_tiny(model_path=None):
    """
    创建一个小型的 Vision Transformer 模型
    Args:
        model_path: 预训练权重的路径，如果为None则不加载预训练权重
    Returns:
        model: 小型 ViT 模型
    """
    model = VisionTransformer(
        img_size=32,
        patch_size=8,
        in_chans=3,
        num_classes=10,
        embed_dim=256,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0
    )
    
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print(f"加载预训练权重: {model_path}")
    
    return model

if __name__ == '__main__':
    import sys
    import os
    # 添加项目根目录到 Python 路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 创建模型实例进行测试
    model = vit_tiny()
    print("\nModel Architecture:")
    print(model)
    
    # 创建随机输入测试前向传播
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
