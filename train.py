import torch
from dataset import HZDataset
import sys
from utils import save_milestone, load_ms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#import config
from config import *
from tqdm import tqdm
from torchvision.utils import save_image
from disc import Disc
from gen import Gen

def training(
        disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
  H_reals = 0
  H_fakes = 0
  loop = tqdm(loader,leave=True)

  for i , (z,h) in enumerate(loop):
    z = z.to(DEVICE)
    h = h.to(DEVICE)

    ## We embark on training the discriminator for horses and zebras first
    with torch.cuda.amp.autocast():

      fake_h = gen_H(z)
      D_H_real = disc_H(h)
      D_H_fake = disc_H(fake_h.detach())
      H_reals += D_H_real.mean().item()
      H_fakes += D_H_fake.mean().item()
      D_H_real_loss = mse(D_H_real,torch.ones_like(D_H_real))
      D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
      D_H_loss = D_H_real_loss + D_H_fake_loss

      fake_z = gen_Z(h)
      D_Z_real = disc_Z(z)
      D_Z_fake = disc_Z(fake_z.detach())
      D_Z_real_loss = mse(D_Z_real,torch.ones_like(D_Z_real))
      D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
      D_Z_loss = D_Z_real_loss + D_Z_fake_loss

      D_loss = (D_H_loss + D_Z_loss) / 2

    opt_disc.zero_grad()
    d_scaler.scale(D_loss).backward()
    d_scaler.step(opt_disc)
    d_scaler.update()

    # Generator part
    with torch.cuda.amp.autocast():
      D_H_fake = disc_H(fake_h)
      D_Z_fake = disc_Z(fake_z)
      loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
      loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

      cycle_z = gen_Z(fake_h)
      cycle_h = gen_H(fake_z)
      cycle_z_loss = l1(z, cycle_z)
      cycle_h_loss = l1(h, cycle_h)

      identity_z = gen_Z(z)
      identity_h = gen_H(h)
      identity_z_loss = l1(z,identity_z)
      identity_h_loss = l1(h,identity_h)

      G_loss = (loss_G_Z + loss_G_H +
                cycle_z_loss * LAMBDA_CYCLE
                + cycle_h_loss * LAMBDA_CYCLE
                + identity_h_loss * LAMBDA_IDENTITY
                + identity_z_loss * LAMBDA_IDENTITY
                )
    opt_gen.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(opt_gen)
    g_scaler.update()

    if i % 200 == 0:
      save_image(fake_h * 0.5 + 0.5, f'saved_images/h_{i}.png')
      save_image(fake_z * 0.5 + 0.5, f'saved_images/z_{i}.png')

    loop.set_postfix(H_real= H_reals / (i+1), H_fake=H_fakes / (i+1))


def main():
  disc_H = Disc(in_chans = 3).to(DEVICE)
  disc_Z = Disc(in_chans = 3).to(DEVICE)

  gen_Z = Gen(img_chans=3,num_residuals=9).to(DEVICE)
  gen_H = Gen(img_chans=3,num_residuals=9).to(DEVICE)
  opt_disc = optim.Adam(
      list(disc_H.parameters()) + list(disc_Z.parameters()),
      lr = LEARNING_RATE,
      betas = (0.5,0.999),
  )
  opt_gen = optim.Adam(
      list(gen_Z.parameters()) + list(gen_H.parameters()),
      lr = LEARNING_RATE,
      betas = (0.5,0.999)
  )
  L1 = nn.L1Loss()
  mse = nn.MSELoss()

  if LOAD_MODEL:
    load_ms(
        MS_GEN_H,gen_H,opt_gen,LEARNING_RATE
    )
    load_ms(
        MS_GEN_Z,gen_Z,opt_gen,LEARNING_RATE
    )
    load_ms(
      MS_CRITIC_H,
      disc_H,
      opt_disc,
      LEARNING_RATE,
    )
    load_ms(
      MS_CRITIC_Z,
      disc_Z,
      opt_disc,
      LEARNING_RATE,
    )
  dataset = HZDataset(
      root_h = 'trainA',#TRAIN_DIR + '/hs',
      root_z = 'trainB', #TRAIN_DIR + '/zs',
      transform=transforms
  )
  val_dataset = HZDataset(
      root_h = 'testA', #"test/hs",
      root_z = 'testB', #"test/zs",
      transform = transforms,
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=1,
      shuffle=False,
      pin_memory=True
  )
  loader = DataLoader(
      dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers = NUM_WORKERS,
      pin_memory=True,
  )
  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  for e in range(NUM_EPOCHS):
    training(
        disc_H,
        disc_Z,
        gen_Z,
        gen_H,
        loader,
        opt_disc,
        opt_gen,
        L1,
        mse,
        d_scaler,
        g_scaler,
    )
  if SAVE_MODEL:
    save_milestone(gen_H,opt_gen,fn=MS_GEN_H)
    save_milestone(gen_Z,opt_gen,fn=MS_GEN_Z)
    save_milestone(disc_H,opt_disc,fn=MS_CRITIC_H)
    save_milestone(disc_Z,opt_disc,fn=MS_CRITIC_Z)

main()

if __name__ == "__main__":
  main()

