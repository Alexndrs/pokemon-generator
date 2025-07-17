import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        '''
        A sinusoidal time embedding layer that generates embeddings based on sine and cosine functions.
        Args:
        embedding_dim (int): Dimension of the output embedding.
        '''
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        '''
        Forward pass to compute the sinusoidal embeddings.
        Args:
        t (Tensor): Input tensor of shape (B, 1) where B is the batch size.
        Returns:
        Tensor: Output tensor of shape (B, embedding_dim).
        '''
        half_dim = self.embedding_dim // 2
        device = t.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Fix: utiliser t.squeeze(-1) pour enlever la dimension supplémentaire
        emb = t.squeeze(-1)[:, None] * emb[None, :]  
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        '''
        A convolutional block with two convolutional layers, group normalization, and GELU activation.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_emb_dim (int): Dimension of the time embedding.
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.GELU()

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        # Add residual connection if dimensions don't match
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        '''
        Forward pass through the convolutional block.
        Args:
        x (Tensor): Input tensor. (B, C, H, W)
        t_emb (Tensor): Time embedding tensor. (B, time_emb_dim)
        Returns:
        Tensor: Output tensor after passing through the block.
        '''
        # Store residual connection
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.conv1(x)
        
        # Add time embedding (B, C) -> (B, C, 1, 1)
        time_emb = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.act1(self.gn1(h))
        
        # Second conv block
        h = self.conv2(h)
        h = self.act2(self.gn2(h))
        
        # Add residual connection
        return h + residual


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        '''
        A U-Net architecture for image processing tasks.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_emb_dim (int): Dimension of the time embedding.
        '''
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Downsampling layers
        self.down1 = ConvBlock(64, 128, time_emb_dim)
        self.down2 = ConvBlock(128, 256, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 256, time_emb_dim)

        # Upsampling layers
        self.up1 = ConvBlock(256 + 256, 128, time_emb_dim)
        self.up2 = ConvBlock(128 + 128, 64, time_emb_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Final conv
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x, t):
        '''
        Forward pass through the U-Net.
        Args:
        x (Tensor): Input tensor of shape (B, C, H, W).
        t (Tensor): Time tensor of shape (B, 1).
        Returns:
        Tensor: Output tensor of shape (B, out_channels, H, W).
        '''
        t_emb = self.time_embedding(t) # (B, time_emb_dim)

        x1 = self.init_conv(x)          # (B, 64, 128, 128)
        d1 = self.down1(x1, t_emb)      # (B, 128, 128, 128)
        p1 = self.pool(d1)              # (B, 128, 64, 64)

        d2 = self.down2(p1, t_emb)      # (B, 256, 64, 64)
        p2 = self.pool(d2)              # (B, 256, 32, 32)

        bn = self.bottleneck(p2, t_emb)  # (B, 256, 32, 32)

        up1 = self.upsample(bn)            # (B, 256, 64, 64)
        up1 = torch.cat([up1, d2], dim=1)  # skip connection (B, 512, 64, 64)
        up1 = self.up1(up1, t_emb)       # (B, 128, 64, 64)

        up2 = self.upsample(up1)         # (B, 128, 128, 128)
        up2 = torch.cat([up2, d1], dim=1) # skip connection (B, 256, 128, 128)
        up2 = self.up2(up2, t_emb)       # (B, 64, 128, 128)

        out = self.final_conv(up2)       # (B, out_channels, 128, 128)

        return out