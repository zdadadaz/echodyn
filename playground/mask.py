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
%matplotlib qt

tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
echonet.config.DATA_DIR = '../../db/EchoNet-Dynamic'
# print(echonet.config.DATA_DIR)
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
kwargs = {"target_type": tasks,
          "mean": mean,
          "std": std
          }
train_dataset = echonet.datasets.Echo(split="train", **kwargs)
