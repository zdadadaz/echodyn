# -*- coding: utf-8 -*-

import torch
import torchvision
import time
import tqdm
import numpy as np
import os
import pathlib
import echonet
import sklearn.metrics
import matplotlib.pyplot as plt
import math

torch.cuda.empty_cache() 

num_epochs=45
modelname="r2plus1d_18"
tasks="EF"
frames=32
period=2
pretrained=True
output=None
device=None
n_train_patients=None
seed=0
num_workers=1
batch_size=8
lr_step_period=None
run_test=False
run_extra_tests=False

echonet.config.DATA_DIR = '../../db/EchoNet-Dynamic'

### Seed RNGs ###
np.random.seed(seed)
torch.manual_seed(seed)

if output is None:
    output = os.path.join("output", "video", "{}_{}_{}_{}".format(modelname, frames, period, "pretrained" if pretrained else "random"))
if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pathlib.Path(output).mkdir(parents=True, exist_ok=True)

model = torchvision.models.video.__dict__[modelname](pretrained=pretrained)

model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.fc.bias.data[0] = 55.6
if device.type == "cuda":
    model = torch.nn.DataParallel(model)
model.to(device)

optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
if lr_step_period is None:
    lr_step_period = math.inf
scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

# image normalization
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

kwargs = {"target_type": tasks,
          "mean": mean,
          "std": std,
          "length": frames,
          "period": period,
          }

checkpoint = torch.load(os.path.join(output, "best.pt"))
model.load_state_dict(checkpoint['state_dict'])
optim.load_state_dict(checkpoint['opt_dict'])
scheduler.load_state_dict(checkpoint['scheduler_dict'])

# Testing
for split in ["val", "test"]:
    ds = echonet.datasets.Echo(split=split, **kwargs, crops="all")
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
    loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, split, None, device, save_all=True, blocks=100)

    with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
        for (filename, pred) in zip(ds.fnames, yhat):
            for (i, p) in enumerate(pred):
                g.write("{},{},{:.4f}\n".format(filename, i, p))
    echonet.utils.latexify()
    yhat = np.array(list(map(lambda x: x.mean(), yhat)))

    fig = plt.figure(figsize=(3, 3))
    lower = min(y.min(), yhat.min())
    upper = max(y.max(), yhat.max())
    plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
    plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
    plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("Actual EF (%)")
    plt.ylabel("Predicted EF (%)")
    plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
    plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
    # plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
    plt.close(fig)

    fig = plt.figure(figsize=(3, 3))
    plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
    for thresh in [35, 40, 45, 50]:
        fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
        print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
        plt.plot(fpr, tpr)

    plt.axis([-0.01, 1.01, -0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
    plt.close(fig)


def run_epoch(model, dataloader, phase, optim, device, save_all=False, blocks=None):

    criterion = torch.nn.MSELoss()  # Standard L2 loss

    runningloss = 0.0

    model.train(phase == 'train')

    counter = 0
    summer = 0
    summer_squared = 0

    yhat = []
    y = []

    with torch.set_grad_enabled(phase == 'train'):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (i, (X, outcome)) in enumerate(dataloader):

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_crops, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                summer += outcome.sum()
                summer_squared += (outcome ** 2).sum()

                if blocks is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + blocks), ...]) for j in range(0, X.shape[0], blocks)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_crops, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = criterion(outputs.view(-1), outcome)

                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                runningloss += loss.item() * X.size(0)
                counter += X.size(0)

                epoch_loss = runningloss / counter

                # str(i, runningloss, epoch_loss,  str(((summer_squared) / counter - (summer / counter)**2).item()))

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(epoch_loss, loss.item(), summer_squared / counter - (summer / counter) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return epoch_loss, yhat, y
