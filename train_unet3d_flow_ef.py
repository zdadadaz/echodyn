# -*- coding: utf-8 -*-

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
from echonet.models.unet3d import UNet3D, UNet3D_ef,UNet3D_ef_separate
from echonet.datasets.echo_3d_flow import Echo3Df
from echonet.datasets.echo_3d_flow_random import Echo3Df_rand
from echonet.datasets.echo_3d_tvflow_random import Echo3Dftv_rand
from echonet.models.deeplabv3 import DeepLabV3_multi_main
from echonet.datasets.echo import Echo
import sklearn.metrics
import torch.nn as nn


# +
def run_epoch(model,modelname, dataloader, phase, optim, device, save_all=False, blocks=None, flag=-1, divide = 2):

    criterion = torch.nn.MSELoss()  # Standard L2 loss

    runningloss = 0.0

    model.train(phase == 'train')

    counter = 0
    summer = 0
    summer_squared = 0

    yhat = []
    y = []
    half_len = int(len(dataloader)/divide)
    if flag == divide:
        endRange = len(dataloader)
    else:
        endRange = half_len*(flag+1)
    frontRange = half_len*flag
    with torch.set_grad_enabled(phase == 'train'):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (i, (X, (outcome, flow), fid)) in enumerate(dataloader):

                if  flag >= 0 and (not (i < endRange and i >= frontRange)):
                    pbar.set_postfix_str("skip, {:.2f}".format(i))
                    pbar.update()
                    continue

#                 flow size 8,2,31,112,112
#                 if blocks is not None:
#                     batch, n_crops, c, f, h, w = flow.shape
#                     flow = flow.permute(3, 0, 1, 2, 4,5)
#                     flow = torch.cat((flow,torch.zeros((1,batch,n_crops, c,h,w), dtype = torch.double)))
#                     flow = flow.permute(1,2,3,0,4,5).type(torch.float)
#                 else:
#                     batch, c, f, h, w = flow.shape
#                     flow = flow.permute(2, 0, 1, 3, 4)
#                     flow = torch.cat((flow,torch.zeros((1,batch,c,h,w), dtype = torch.double)))
#                     flow = flow.permute(1,2,0,3,4).type(torch.float)
                
                y.append(outcome.numpy())
                X = X.to(device)
                flow = flow.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_crops, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)
                    batch, n_crops, c, f, h, w = flow.shape
                    flow = flow.view(-1, c, f, h, w)

                summer += outcome.sum()
                summer_squared += (outcome ** 2).sum()

                if 'flow' in modelname.split('_'):
                    if blocks is None:
                        outputs = model(flow)
                    else:
                        outputs = torch.cat([model(flow[j:(j + blocks), ...]) for j in range(0, flow.shape[0], blocks)])
                else:
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


# -

#                 print(outputs.view(-1))
#                 print(outcome)
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


# -

def run(num_epochs=45,
        modelname="r3d_18",
        tasks="EF",
        frames=16,
        period=4,
        pretrained=True,
        output=None,
        device=None,
        n_train_patients=None,
        seed=0,
        num_workers=4,
        batch_size=20,
        lr_step_period=None,
        run_test=False,
        run_extra_tests=False):

    ### Seed RNGs ###
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    tasks = ['EF','flow']
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(modelname, frames, period, "pretrained" if pretrained else "random"))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    if "unet3d" in modelname.split('_'):
        if "separate" in modelname.split('_'):
            model = UNet3D_ef_separate(in_channels=3, out_channels=1)
        else:
            model = UNet3D_ef(in_channels=3, out_channels=1)
    else:
        if "flow" in modelname.split('_'): # input feature size = 2
            model = torchvision.models.video.__dict__['r2plus1d_18'](pretrained=pretrained)
            model.stem[0] = nn.Conv3d(2, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        else: # input feature size = 3
            model = torchvision.models.video.__dict__['r2plus1d_18'](pretrained=pretrained)
    
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

# Data preparation
    train_dataset = Echo3Dftv_rand(split="train", **kwargs)
#     train_dataset = Echo3Df_rand(split="train", **kwargs)
    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataset = Echo3Dftv_rand(split="val", **kwargs)
#     val_dataset = Echo3Df_rand(split="val", **kwargs)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        # read previous trained model
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        # Train one epoch
        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)
                loss, yhat, y = run_epoch(model, modelname, dataloaders[phase], phase, optim, device)
                
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(yhat, y),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(yhat, y),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()


        # Testing
        if run_test:
            for split in ["val", "test"]:
                dataset = Echo3Dftv_rand(split=split, **kwargs)
#                 dataset = Echo3Df_rand(split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                loss, yhat, y = run_epoch(model,modelname, dataloader, split, None, device)
                f.write("{} (one crop) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one crop) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one crop) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                ds = Echo3Dftv_rand(split=split, **kwargs, crops="all")
#                 ds = Echo3Df_rand(split=split, **kwargs, crops="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                yhat = []
                y = []
                divide= 5
                for d in range(divide):
                    loss, yhat1, y1 = run_epoch(model,modelname, dataloader, split, None, device, save_all=True, blocks=50, flag = d, divide = divide)
                    yhat.append(yhat1)
                    y.append(y1)
#                 loss, yhat1, y1 = run_epoch(model,modelname, dataloader, split, None, device, save_all=True, blocks=50, flag = 1)
#                 yhat.append(yhat1)
#                 y.append(y1)
                yhat = np.concatenate(yhat)
                y = np.concatenate(y)
                f.write("{} (all crops) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all crops) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all crops) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

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

echonet.config.DATA_DIR = '../../data/EchoNet-Dynamic'

run(modelname="r2plus1_ef_flow_tv",
        frames=33,
        period=2,
        pretrained=False,
        batch_size=8,
        run_test=True,
        num_epochs = 50)

# +
# run(modelname="r2plus1_ef",
#         frames=32,
#         period=2,
#         pretrained=True,
#         batch_size=8,
#         run_test=True,
#         num_epochs = 50)

# +

# run(modelname="unet3d_ef_flow_xy_gray",
#         frames=32,
#         period=2,
#         pretrained=False,
#         batch_size=8,
#         run_test=True,
#         num_epochs = 50)
