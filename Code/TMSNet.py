import torch
import torch.nn as nn
from collections.abc import Sequence
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat
from monai.utils import ensure_tuple_rep
from Code.Transformer_cross import TransformerCrossModel
from Code.PositionalEncoding import LearnedPositionalEncoding

class TMSNet(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        embedding_dim: int = 192,
        seq_length_H: int = 3240,
        seq_length_L: int = 4096,
        in_channels_H: int = 1,
        in_channels_L: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (16, 32, 64, 128, 256, 192, 128, 64, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ): 
    
        super().__init__()
        fea = ensure_tuple_rep(features, 9)
        # print(f"TMSUNet features: {fea}.")
        
        self.conv_0_H = TwoConv(spatial_dims, in_channels_H, fea[0], act, norm, bias, dropout)
        self.down_1_H = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2_H = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3_H = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4_H = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.upcat_4_H = UpCat(spatial_dims, fea[4], fea[3], fea[5], act, norm, bias, dropout, upsample)
        
        self.conv_0_L = TwoConv(spatial_dims,in_channels_L,fea[0], act, norm, bias, dropout)
        self.down_1_L = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2_L = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3_L = Down(spatial_dims, fea[2], fea[5], act, norm, bias, dropout)

        self.position_encoding_H = LearnedPositionalEncoding(embedding_dim, seq_length_H)
        self.position_encoding_L = LearnedPositionalEncoding(embedding_dim, seq_length_L)
        
        self.transformer_cross = TransformerCrossModel(fea[5],1,4,fea[5] *2,0.1,0.1)
        
        self.upcat_3_H = UpCat(spatial_dims, fea[5], fea[2], fea[6], act, norm, bias, dropout, upsample)
        self.upcat_2_H = UpCat(spatial_dims, fea[6], fea[1], fea[7], act, norm, bias, dropout, upsample)
        self.upcat_1_H = UpCat(spatial_dims, fea[7], fea[0], fea[8], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[8], out_channels, kernel_size=1)
        
    def forward(self, x_H: torch.Tensor, x_L: torch.Tensor):
    
        x0_H = self.conv_0_H(x_H)
        x1_H = self.down_1_H(x0_H)
        x2_H = self.down_2_H(x1_H)
        x3_H = self.down_3_H(x2_H)
        x4_H = self.down_4_H(x3_H)
        u4_H = self.upcat_4_H(x4_H, x3_H)

        x0_L = self.conv_0_L(x_L)
        x1_L = self.down_1_L(x0_L)
        x2_L = self.down_2_L(x1_L)
        x3_L = self.down_3_L(x2_L)
        
        B, C, H, W, D = u4_H.shape
        u4_H = u4_H.view(B, C, -1).permute(0, 2, 1)
        u4_H = self.position_encoding_H(u4_H)
        
        x3_L = x3_L.view(B, C, -1).permute(0, 2, 1)
        x3_L = self.position_encoding_L(x3_L)
        
        u4_fused = self.transformer_cross(u4_H, x3_L)
        u4_fused = u4_fused.permute(0, 2, 1).view(B, C, H, W, D)

        u3_H = self.upcat_3_H(u4_fused, x2_H)
        u2_H = self.upcat_2_H(u3_H, x1_H)
        u1_H = self.upcat_1_H(u2_H, x0_H)
        
        logits = self.final_conv(u1_H)
        return logits
