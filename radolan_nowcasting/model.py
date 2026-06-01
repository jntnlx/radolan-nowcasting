"""

Residual U-Net.

Predicts last-frame persistence departure: if network outputs only zeros, 
forecast equals to most recent observation repeated across all lead times.  
Residual formulation provides strong prior, i.e. model only needs to learn 
changes from persistence instead of full precipitation field from scratch.

Architecture: 
  5-level encoder-decoder with skip connections.
  - Encoder: repeated (Conv3x3 → BN → ReLU) blocks with MaxPool
  - Bottleneck: same block structure
  - Decoder: bilinear upsampling → skip-cat → conv blocks
  - Output: 1x1 conv projecting to SEQ_LEN_OUT channels + residual add

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SEQ_LEN_IN, SEQ_LEN_OUT


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Bilinear upsample → concatenate skip → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    5-level residual U-Net: 6 input frames → 6 forecast frames.

    Residual connection adds persistence of last frame to network output.
    Model learns corrections instead of absolute fields!
    """

    def __init__(
        self,
        in_channels: int = SEQ_LEN_IN,
        out_channels: int = SEQ_LEN_OUT,
        base_features: int = 64,
    ):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder
        self.dec4 = UpsampleBlock(f * 16, f * 8, f * 8)
        self.dec3 = UpsampleBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpsampleBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpsampleBlock(f * 2, f, f)

        self.head = nn.Conv2d(f, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial dimension must be divisible by 2^4 (16)
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"Spatial dims must be divisible by 16, got ({h}, {w})"
            )

        # Last observed frame as persistence baseline
        last_frame = x[:, -1:, :, :]

        # Encoder path
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))

        # Decoder path with skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Residual: network output + persistence
        return self.head(d1) + last_frame