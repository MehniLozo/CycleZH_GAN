import torch
import torch.nn as nn

class Disc_block(nn.Module):
  def __init__(self,in_chans, out_chans, stride):
    super().__init__()
    # Convolution-InstanceNorm-LeakyReLU -> Ck -> k filters & stride 2
    self.conv = nn.Sequential(
        nn.Conv2d(
            in_chans,
            out_chans,
            4,
            stride,
            1,
            bias=True,
            padding_mode="reflect",
        ),
        nn.InstanceNorm2d(out_chans),
        nn.LeakyRelU(0.2,inplace=True),
    )
  def forward(self,x):
    return self.conv(x)
