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