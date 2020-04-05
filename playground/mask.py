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

tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "EF", "ESV", "EDV"]
echonet.config.DATA_DIR = '../../db/EchoNet-Dynamic'
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

train_dataset = echonet.datasets.Echo(split="all", target_type=["Filename", "LargeIndex", "SmallIndex"], length=None, period=1, segmentation=os.path.join(output, "labels"))
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=False)
