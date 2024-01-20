import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_chans,out_chans, down= True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans,**kwargs) if down 
            else nn.ConvTranspose2d(in_chans,out_chans,**kwargs),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )
    def forward(self,x):
        return self.conv(x)
      
class ResBlock(nn.Module):
  def __init__(self,chans):
    super().__init()
    self.block = nn.Sequential(
        ConvBlock(chans, chans, kernel_size= 2, padding=1),
        ConvBlock(chans, chans, use_act=False,kernel_size=3,padding=1),
    )
    def forward(self,x):
      return x + self.block(x)


class Gen(nn.Module):
  def __init__(self,img_chans,num_features=64,num_residuals=9):
    super().__init__()
    self.initial = nn.Sequential(
        nn.Conv2d(
            img_chans,
            num_features,
            kernel_size=7,
            stride = 1,
            padding=3,
            padding_mode="reflect"
        ),
        nn.InstanceNorm2d(num_features),
        nn.ReLU(inplace=True)
    )
    self.downbs = nn.ModuleList(
        [
            ConvBlock(
                num_features,num_features*2,kernel_size=3,
                stride=2,padding=1
            ),
            ConvBlock(
                num_features * 2,
                num_features * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        ]
    )
    self.resbs = nn.Sequential(
        *[ResBlock(num_features * 4) for _ in range(num_residuals)]
    )
    self.upbs = nn.ModuleList(
        [
            ConvBlock(
                num_features * 4,
                num_features * 2,
                down = False,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding=1,
            ),
            ConvBlock(
                num_features * 2,
                num_features * 1,
                down=False,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding=1
            ),
        ]
    )
    self.end = nn.Conv2d(
        num_features * 1,
        img_chans,
        kernel_size=7,
        stride = 1,
        padding=3,
        padding_mode="reflect",
    )
  def forward(self,x):
    x = self.initial(x)
    for layer in self.downbs:
      x = layer(x)
    x = self.resbs(x)
    for layer in self.upbs:
      x = layer(x)
    return torch.tanh(self.end(x))
