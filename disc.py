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


class Discriminator(nn.Module):
  def __init__(self,in_chans = 3,features = [64,128,256,512]):
    super().__init__()
    self.initial = nn.Sequential(
        nn.Conv2d(
            in_chans,
            features[0],
            kernel_size = 4,
            stride = 2,
            padding = 1,
            padding_mode = "reflect"
        ),
        nn.LeakyReLU(0.2,inplace=True),
    )
    layers = []
    in_chans = features[0]
    for ft in features[1:]:
      layers.append(
          Disc_block(in_chans, ft, stride = 1 if ft == features[-1] else 2)
      )
      in_chans = ft
      layers.append(
          nn.Conv2d(
              in_chans,
              1,
              kernel_size = 4,
              stride = 1,
              padding = 1,
              padding_mode = "reflect"
          )
      )
      self.model = nn.Sequential(*layers)
  def forward(self,x):
    x = self.initial(x)
    return torch.sigmoid(self.model(x))