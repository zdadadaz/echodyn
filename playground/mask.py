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

torch.cuda.empty_cache() 
tasks = ["EF"]
echonet.config.DATA_DIR = '../../../data/EchoNet-Dynamic'
# print(echonet.config.DATA_DIR)
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
kwargs = {"target_type": tasks,
          "mean": mean,
          "std": std
          }
train_dataset = echonet.datasets.Echo(split="train", **kwargs)

num_workers=5
batch_size=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

# pos =0
# for (i, (_, (large_frame, small_frame, large_trace, small_trace, ef, esv, edv))) in enumerate(dataloader):
#     pos += (large_trace == 1).sum().item()
#     print((large_trace == 1).sum().item())
#     pos += (small_trace == 1).sum().item()

# count = 0
# with tqdm.tqdm(total=10) as pbar:
#     for (i, (_, (large_frame, small_frame, large_trace, small_trace, ef, esv, edv))) in enumerate(dataloader):
#         if count == 10:
#             break
#         pbar.update()
#         count += 1

frames=32
period=2
kwargs = {"target_type": ["LargeTrace","SmallTrace","EF"],
          "mean": mean,
          "std": std,
          "length": frames,
          "period": period,
          }

ds = echonet.datasets.Echo(split="test", **kwargs, crops="all")
dataloader = torch.utils.data.DataLoader(
    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

train_dataset = Echo3D(split="val", **kwargs, pad=12)

# train_dataset = echonet.datasets.Echo(split="all", target_type=["LargeIndex","SmallIndex","EF"], length=None, period=1, segmentation=os.path.join(output, "labels"))
# dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=1, shuffle=False, pin_memory=False)
pos =0
for (i, (X, outcome, fid) ) in enumerate(dataloader):
    print(fid)
    pos +=1

    



    

    

    

    

    

    

    
