# David Ouyang 12/5/2019

# Notebook which:
# 1. Downloads weights
# 2. Initializes model and imports weights
# 3. Performs test time evaluation of videos (already preprocessed with ConvertDICOMToAVI.ipynb)

import re
import os, os.path
from os.path import splitext
# import pydicom as dicom
import numpy as np
# from pydicom.uid import UID, generate_uid
import shutil
from multiprocessing import dummy as multiprocessing
import time
import subprocess
import datetime
from datetime import date
import sys
import cv2
import matplotlib.pyplot as plt
import sys
from shutil import copy
import math
import torch
import torchvision

sys.path.append("..")
import echonet

import wget 

#destinationFolder = "/Users/davidouyang/Dropbox/Echo Research/CodeBase/Output"
destinationFolder = "./../provided_Output"
#videosFolder = "/Users/davidouyang/Dropbox/Echo Research/CodeBase/a4c-video-dir"
videosFolder = "./../../../data/EchoNet-Dynamic/Videos"
#DestinationForWeights = "/Users/davidouyang/Dropbox/Echo Research/CodeBase/EchoNetDynamic-Weights"
DestinationForWeights = "./../weights/"

# Download model weights

if os.path.exists(DestinationForWeights):
    print("The weights are at", DestinationForWeights)
else:
    print("Creating folder at ", DestinationForWeights, " to store weights")
    os.mkdir(DestinationForWeights)

segmentationWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt'
ejectionFractionWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt'


if not os.path.exists(os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL))):
    print("Downloading Segmentation Weights, ", segmentationWeightsURL," to ",os.path.join(DestinationForWeights,os.path.basename(segmentationWeightsURL)))
    filename = wget.download(segmentationWeightsURL, out = DestinationForWeights)
else:
    print("Segmentation Weights already present")

if not os.path.exists(os.path.join(DestinationForWeights, os.path.basename(ejectionFractionWeightsURL))):
    print("Downloading EF Weights, ", ejectionFractionWeightsURL," to ",os.path.join(DestinationForWeights,os.path.basename(ejectionFractionWeightsURL)))
    filename = wget.download(ejectionFractionWeightsURL, out = DestinationForWeights)
else:
    print("EF Weights already present")


# EFFFFF
# # Initialize and Run EF model

frames = 32
period = 1 #2
batch_size = 20
model = torchvision.models.video.r2plus1d_18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)



print("loading weights from ", os.path.join(DestinationForWeights, "r2plus1d_18_32_2_pretrained"))

if torch.cuda.is_available():
    print("cuda is available, original weights")
    device = torch.device("cuda")
    model = torch.nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(ejectionFractionWeightsURL)))
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("cuda is not available, cpu weights")
    device = torch.device("cpu")
    checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(ejectionFractionWeightsURL)), map_location = "cpu")
    state_dict_cpu = {k[7:]: v for (k, v) in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict_cpu)


# try some random weights: final_r2+1d_model_regression_EF_sgd_skip1_32frames.pth.tar
# scp ouyangd@arthur2:~/Echo-Tracing-Analysis/final_r2+1d_model_regression_EF_sgd_skip1_32frames.pth.tar "C:\Users\Windows\Dropbox\Echo Research\CodeBase\EchoNetDynamic-Weights"
# Weights = "final_r2+1d_model_regression_EF_sgd_skip1_32frames.pth.tar"


output = os.path.join(destinationFolder, "cedars_ef_output.csv")

ds = echonet.datasets.Echo(split = "external_test", external_test_location = videosFolder, crops="all")
print(ds.split, ds.fnames)

mean, std = echonet.utils.get_mean_and_std(ds)

kwargs = {"target_type": "EF",
          "mean": mean,
          "std": std,
          "length": frames,
          "period": period,
          }

ds = echonet.datasets.Echo(split = "external_test", external_test_location = videosFolder, **kwargs, crops="all")

test_dataloader = torch.utils.data.DataLoader(ds, batch_size = 1, num_workers = 5, shuffle = True, pin_memory=(device.type == "cuda"))
loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, "test", None, device, save_all=True, blocks=25)

with open(output, "w") as g:
    for (filename, pred) in zip(ds.fnames, yhat):
        for (i,p) in enumerate(pred):
            g.write("{},{},{:.4f}\n".format(filename, i, p))

# Segmentation
# # # Initialize and Run Segmentation model

torch.cuda.empty_cache()


videosFolder = "C:\\Users\\Windows\\Dropbox\\Echo Research\\CodeBase\\View Classification\\AppearsA4c\\Resized2"

def collate_fn(x):
    x, f = zip(*x)
    i = list(map(lambda t: t.shape[1], x))
    x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
    return x, f, i

dataloader = torch.utils.data.DataLoader(echonet.datasets.Echo(split="external_test", external_test_location = videosFolder, target_type=["Filename"], length=None, period=1, mean=mean, std=std),
                                         batch_size=10, num_workers=0, shuffle=False, pin_memory=(device.type == "cuda"), collate_fn=collate_fn)
if not all([os.path.isfile(os.path.join(destinationFolder, "labels", os.path.splitext(f)[0] + ".npy")) for f in dataloader.dataset.fnames]):
    # Save segmentations for all frames
    # Only run if missing files

    pathlib.Path(os.path.join(destinationFolder, "labels")).mkdir(parents=True, exist_ok=True)
    block = 1024
    model.eval()

    with torch.no_grad():
        for (x, f, i) in tqdm.tqdm(dataloader):
            x = x.to(device)
            y = np.concatenate([model(x[i:(i + block), :, :, :])["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block)]).astype(np.float16)
            start = 0
            for (filename, offset) in zip(f, i):
                np.save(os.path.join(destinationFolder, "labels", os.path.splitext(filename)[0]), y[start:(start + offset), 0, :, :])
                start += offset

dataloader = torch.utils.data.DataLoader(echonet.datasets.Echo(split="external_test", external_test_location = videosFolder, target_type=["Filename"], length=None, period=1, segmentation=os.path.join(destinationFolder, "labels")),
                                         batch_size=1, num_workers=8, shuffle=False, pin_memory=False)
if not all(os.path.isfile(os.path.join(destinationFolder, "videos", f)) for f in dataloader.dataset.fnames):
    pathlib.Path(os.path.join(destinationFolder, "videos")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(destinationFolder, "size")).mkdir(parents=True, exist_ok=True)
    echonet.utils.latexify()
    with open(os.path.join(destinationFolder, "size.csv"), "w") as g:
        g.write("Filename,Frame,Size,ComputerSmall\n")
        for (x, filename) in tqdm.tqdm(dataloader):
            x = x.numpy()
            for i in range(len(filename)):
                img = x[i, :, :, :, :].copy()
                logit = img[2, :, :, :].copy()
                img[1, :, :, :] = img[0, :, :, :]
                img[2, :, :, :] = img[0, :, :, :]
                img = np.concatenate((img, img), 3)
                img[0, :, :, 112:] = np.maximum(255. * (logit > 0), img[0, :, :, 112:])

                img = np.concatenate((img, np.zeros_like(img)), 2)
                size = (logit > 0).sum(2).sum(1)
                try:
                    trim_min = sorted(size)[round(len(size) ** 0.05)]
                except:
                    import code; code.interact(local=dict(globals(), **locals()))
                trim_max = sorted(size)[round(len(size) ** 0.95)]
                trim_range = trim_max - trim_min
                peaks = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
                for (x, y) in enumerate(size):
                    g.write("{},{},{},{}\n".format(filename[0], x, y, 1 if x in peaks else 0))
                fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                ylim = plt.ylim()
                for p in peaks:
                    plt.plot(np.array([p, p]) / 50, ylim, linewidth=1)
                plt.ylim(ylim)
                plt.title(os.path.splitext(filename[i])[0])
                plt.xlabel("Seconds")
                plt.ylabel("Size (pixels)")
                plt.tight_layout()
                plt.savefig(os.path.join(destinationFolder, "size", os.path.splitext(filename[i])[0] + ".pdf"))
                plt.close(fig)
                size -= size.min()
                size = size / size.max()
                size = 1 - size
                for (x, y) in enumerate(size):
                    img[:, :, int(round(115 + 100 * y)), int(round(x / len(size) * 200 + 10))] = 255.
                    interval = np.array([-3, -2, -1, 0, 1, 2, 3])
                    for a in interval:
                        for b in interval:
                            img[:, x, a + int(round(115 + 100 * y)), b + int(round(x / len(size) * 200 + 10))] = 255.
                    if x in peaks:
                        img[:, :, 200:225, b + int(round(x / len(size) * 200 + 10))] = 255.
                echonet.utils.savevideo(os.path.join(destinationFolder, "videos", filename[i]), img.astype(np.uint8), 50)                
