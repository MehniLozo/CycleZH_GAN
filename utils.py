import random , torch, os, numpy as np
import torch.nn as nn 
import config
import copy 

def save_milestone(model,optimizer, fn="ms.pth.tar"):
    print("----> Saving Milestone")
    ms = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(ms, fn)

def load_ms(fn, model, optimizer, lr):
    print("---> Loading ...")
    ms = torch.load(fn, map_location = config.DEVICE)
    model.load_state_dict(ms["state_dict"])
    optimizer.load_state_dict(ms["optimizer"])

    for p_group in optimizer.param_groups:
        p_group['lr'] = lr # we change the learning rate

def seeding(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.becnhmark = False