import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from nets.vit import VisionTransformer

# 定义一个使用剪枝和压缩技术的ViT模型
class PrunedVisionTransformer(VisionTransformer):
    def __init__(self, img_size=32, patch_size=16, in_chans=3, num_classes=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.):
        super(PrunedVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            mlp_head=mlp_head,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate
        )
        
        # 对模型的线性层进行剪枝
        self.apply_pruning()

    def apply_pruning(self):
        # 对每个Transformer block中的MLP层进行剪枝
        for block in self.blocks:
            # 检查并访问 MLP 层的全连接层
            if isinstance(block.mlp.fc1, nn.Linear):
                prune.l1_unstructured(block.mlp.fc1, name='weight', amount=0.4)
            if isinstance(block.mlp.fc2, nn.Linear):
                prune.l1_unstructured(block.mlp.fc2, name='weight', amount=0.4)

    def forward(self, x):
        # 直接使用父类的forward方法
        return super().forward(x)

def vit_pruned(model_path=None):
    """
    创建一个使用剪枝技术的ViT模型
    Args:
        model_path: 预训练权重的路径，如果为None则不加载预训练权重
    Returns:
        model: 剪枝后的ViT模型
    """
    model = PrunedVisionTransformer(
        img_size=32,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_head=False,
        drop_rate=0.,
        attn_drop_rate=0.
    )
    
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    
    return model

if __name__ == '__main__':
    model = vit_pruned()
    print(model)
