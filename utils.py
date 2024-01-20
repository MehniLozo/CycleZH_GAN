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