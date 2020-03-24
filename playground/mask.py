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

tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
echonet.config.DATA_DIR = '../../db/EchoNet-Dynamic'
# print(echonet.config.DATA_DIR)
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
kwargs = {"target_type": tasks,
          "mean": mean,
          "std": std
          }
train_dataset = echonet.datasets.Echo(split="train", **kwargs)

num_workers=5
batch_size=20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

# pos =0
# for (i, (_, (large_frame, small_frame, large_trace, small_trace))) in enumerate(dataloader):
#     pos += (large_trace == 1).sum().item()
#     print((large_trace == 1).sum().item())
#     pos += (small_trace == 1).sum().item()

# aa = train_dataset.__getitem__(0)
# len(aa[1])

# plt.imshow(aa[1][0].transpose(1,2,0))
# plt.savefig("test1.png")

# plt.imshow(aa[1][1].transpose(1,2,0))
# plt.savefig("test2.png")

# plt.imshow(aa[1][2])
# plt.savefig("test3.png")

# plt.imshow(aa[1][3])
# plt.savefig("test4.png")

# plt.imshow(aa[0].transpose(1,2,3,0)[0])
# plt.savefig("org1.png")

# def collate_fn(x):
#     x, f = zip(*x)
#     i = list(map(lambda t: t.shape[1], x))
#     x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
#     return x, f, i
# dataloader = torch.utils.data.DataLoader(echonet.datasets.Echo(split="all", target_type=["Filename"], length=None, period=1, mean=mean, std=std),
#                                              batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), collate_fn=collate_fn)
# block = 1024
# for (x, f, i) in tqdm.tqdm(dataloader):
#     x = x.to(device)

tasks="EF"
frames=32
period=2
kwargs = {"target_type": tasks,
          "mean": mean,
          "std": std,
          "length": frames,
          "period": period,
          }

# Data preparation
train_dataset = echonet.datasets.Echo(split="train", **kwargs, pad=12)
aa = train_dataset.__getitem__(0)
# dataloader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
# y=[]
# for (i, (X, outcome)) in enumerate(dataloader):

#     y.append(outcome.numpy())
#     X = X.to(device)
#     outcome = outcome.to(device)