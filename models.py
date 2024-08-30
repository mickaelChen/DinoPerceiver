# Modified from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.jit import Final

import timm

from timm.models.vision_transformer import Mlp, VisionTransformer
from timm.layers import use_fused_attn

class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, Nx, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, Nx, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1(y))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# Alternative forward function to compute CA instead of SA for a SA block
def ca_forward(attn: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, Nx, C = x.shape
    Ny = y.shape[1]
    size = attn.qkv.weight.shape[0] // 3
    
    q = F.linear(x, attn.qkv.weight[:size], attn.qkv.bias[:size])
    q = q.reshape(B, Nx, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
    
    kv = F.linear(y, attn.qkv.weight[size:], attn.qkv.bias[size:])
    kv = kv.reshape(B, Ny, 2, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
            
    q, k = attn.q_norm(q), attn.k_norm(k)
    
    if attn.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=attn.attn_drop.p if attn.training else 0.,
        )
    else:
        q = q * attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, Nx, C)
    x = attn.proj(x)
    x = attn.proj_drop(x)
    return x

# Alternative forward function to compute CA instead of SA for a pretrained SA block
def ca_block_forward(block: nn.Module, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    q = q + block.drop_path1(block.ls1(ca_forward(block.attn, block.norm1(q), block.norm1(kv))))
    q = q + block.drop_path2(block.ls2(block.mlp(block.norm2(q))))
    return q

class DinoPerceiver(nn.Module):
    def __init__(self, n_tokens, dino_size='base',
                 ca_depth=1, ca_mlp_ratio=4.0):
        super().__init__()

        self.n_tokens = n_tokens
        self.dino = timm.create_model(f'vit_{dino_size}_patch14_reg4_dinov2.lvd142m', pretrained=True)
        self.queries = nn.Parameter(torch.randn((1, n_tokens, self.dino.num_features))) # Initialize them !
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(dim=self.dino.num_features,
                                                            num_heads=self.dino.blocks[0].attn.num_heads,
                                                            mlp_ratio=ca_mlp_ratio)
                                        for i in range(ca_depth)])

        # Freeze dino
        for param in self.dino.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINO forward_features from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        x = self.dino.patch_embed(x)
        x = self.dino._pos_embed(x)
        x = self.dino.patch_drop(x)  # no-op
        x = self.dino.norm_pre(x)    # no-op
        
        ### prepare latent queries
        perceiver_tokens = self.queries.repeat(x.shape[0], 1, 1)

        # Unroll blocks.forward(x)
        # if self.dino.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.dino.blocks(x)
        for block in self.dino.blocks:
            perceiver_tokens = ca_block_forward(block, perceiver_tokens, x)
            x = block(x)
        
        x = self.dino.norm(x)                               # layer norm
        perceiver_tokens = self.dino.norm(perceiver_tokens) # layer norm
        
        for block in self.ca_blocks:
            perceiver_tokens = block(perceiver_tokens, x)

        return x, perceiver_tokens

class DinoPerceiver_v1(nn.Module):
    def __init__(self, n_tokens, dino_size='base',
                 ca_depth=1, ca_mlp_ratio=4.0):
        super().__init__()

        self.n_tokens = n_tokens
        self.dino = timm.create_model(f'vit_{dino_size}_patch14_reg4_dinov2.lvd142m')
        self.queries = nn.Parameter(torch.randn((1, n_tokens, self.dino.num_features))) # Initialize them !
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(dim=self.dino.num_features,
                                                            num_heads=self.dino.blocks[0].attn.num_heads,
                                                            mlp_ratio=ca_mlp_ratio)
                                        for i in range(ca_depth)])
        
        # Freeze dino
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINO forward_features from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        x = self.dino.patch_embed(x)
        x = self.dino._pos_embed(x)        
        x = self.dino.patch_drop(x)  # no-op

        ### Add latent queries here
        x = torch.cat((self.queries.repeat(x.shape[0], 1, 1), x), 1)
        
        x = self.dino.norm_pre(x)    # no-op
        # if self.dino.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.dino.blocks(x)
        x = self.dino.blocks(x)
        x = self.dino.norm(x)        # layer norm

        y = x[:, :self.n_tokens]
        for block in self.ca_blocks:
            y = block(y, x)
        
        return x[:, self.n_tokens:], y


if __name__ == "__main__":
    model = DinoPerceiver(n_tokens=4, dino_size='base', ca_depth=1)
    model_v1 = DinoPerceiver_v1(n_tokens=4, dino_size='base', ca_depth=1)

    batch_size = 2
    img = torch.randn(batch_size, 3, model.dino.patch_embed.img_size[0], model.dino.patch_embed.img_size[1])
    with torch.no_grad():
        dino_features, perceiver_features = model(img)
        dino_features_v1, perceiver_features_v1 = model_v1(img)
        orig_dino_features = model.dino.forward_features(img)
        
    print("Output shapes:", dino_features.shape, perceiver_features.shape)
    print("Max diff between augmented dino and orig dino", (dino_features - orig_dino_features).abs().max().cpu().item())
    print("Max diff between augmented dino v1 and orig dino", (dino_features_v1 - orig_dino_features).abs().max().cpu().item())
    
    
