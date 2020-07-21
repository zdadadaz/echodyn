import numpy as np
import matplotlib.pyplot as plt
import math
import echonet
import torch
import os
import torchvision
import pathlib
import tqdm
import scipy.signal
import time
import matplotlib.pyplot as plt
# %matplotlib qt
from echonet.datasets.echo_3d import Echo3D
from echonet.datasets.echo_3d_flow_random_seg import Echo3Df_rand_seg

torch.cuda.empty_cache() 
# tasks = ["EF"]
echonet.config.DATA_DIR = '../../data/EchoNet-Dynamic'
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

num_workers=5
batch_size=8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


frames=32
period=2
kwargs = {"target_type": ["EF"],
          "mean": mean,
          "std": std,
          "length": frames,
          "period": period,
          }

ds = Echo3Df_rand_seg(split="train", **kwargs, crops=2)
dataloader = torch.utils.data.DataLoader(
    ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

pos =0
for (i, (X, outcome, fid) ) in enumerate(dataloader):
    print("X",X.size())
#     pos +=1

    



    

    

    

    

    

    

    
