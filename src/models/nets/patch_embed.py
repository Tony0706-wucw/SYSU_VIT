import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # (B, C, H, W) -> (B, D, H//P, W//P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        
        return x
