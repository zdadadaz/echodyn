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
from echonet.models.unet3d import UNet3D, UNet3D_ef,UNet3D_ef_separate,UNet3D_ef_esv_edv
# from echonet.models.s3d import model_s3d
from echonet.models.pre_resnet2p1d import generate_model as gen_r2p1d
# from echonet.models.resnet3d import resnet50
# from echonet.models.deeplabv3 import DeepLabV3_multi_main
from echonet.models.tsn import TSN_r2p1_18
from echonet.datasets.echo import Echo
import sklearn.metrics


def run_epoch(model, dataloader, phase, optim, device, save_all=False, blocks=None, flag=-1, divide = 2):

    criterion = torch.nn.MSELoss()  # Standard L2 loss

    runningloss = 0.0
    runningloss_ef = 0.0
    runningloss_esv = 0.0
    runningloss_edv = 0.0
    
    if phase=='train':
        model.train(phase == 'train')
    else:
        model.eval()
        
    counter = 0
    summer = 0
    summer_squared = 0

    yhat = []
    yhat_esv = []
    yhat_edv = []
    y = []
    y_esv = []
    y_edv = []
    half_len = int(len(dataloader)/divide)
    if flag == divide-1:
        endRange = len(dataloader)
    else:
        endRange = half_len*(flag+1)
    frontRange = half_len*flag
    with torch.set_grad_enabled(phase == 'train'):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (i, (X, (ef,esv,edv))) in enumerate(dataloader):

                if  flag >= 0 and (not (i < endRange and i >= frontRange)):
                    pbar.set_postfix_str("skip, {:.2f}".format(i))
                    pbar.update()
                    continue
                y.append(ef.numpy())
                y_esv.append(esv.numpy())
                y_edv.append(edv.numpy())
                
                X = X.to(device)
                ef = ef.to(device)
                esv = esv.to(device)
                edv = edv.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_crops, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                summer += ef.sum()
                summer_squared += (ef ** 2).sum()

                if blocks is None:
                    ef_out,esv_out,edv_out = model(X)
                else:
                    tmp = torch.cat([model(X[j:(j + blocks), ...]) for j in range(0, X.shape[0], blocks)])
                    ef_out = torch.cat([tmp[j][0] for j in range(tmp.shape[0])])
                    esv_out = torch.cat([tmp[j][1] for j in range(tmp.shape[0])])
                    edv_out = torch.cat([tmp[j][2] for j in range(tmp.shape[0])])

                if save_all:
                    yhat.append(ef_out.view(-1).to("cpu").detach().numpy())
                    yhat_esv.append(esv_out.view(-1).to("cpu").detach().numpy())
                    yhat_edv.append(edv_out.view(-1).to("cpu").detach().numpy())

                if average:
                    ef_out = ef_out.view(batch, n_crops, -1).mean(1)
                    esv_out = esv_out.view(batch, n_crops, -1).mean(1)
                    edv_out = edv_out.view(batch, n_crops, -1).mean(1)
                    
                if not save_all:
                    yhat.append(ef_out.view(-1).to("cpu").detach().numpy())
                    yhat_esv.append(esv_out.view(-1).to("cpu").detach().numpy())
                    yhat_edv.append(edv_out.view(-1).to("cpu").detach().numpy())

#                 print(outputs.view(-1))
#                 print(outcome)
                loss_ef = criterion(ef_out.view(-1), ef)
                loss_esv = criterion(esv_out.view(-1), esv)
                loss_edv = criterion(edv_out.view(-1), edv)
                
                loss = loss_ef + loss_esv + loss_edv
                
                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                runningloss += loss.item() * X.size(0)
                runningloss_ef += loss_ef.item() * X.size(0)
                runningloss_esv += loss_esv.item() * X.size(0)
                runningloss_edv += loss_edv.item() * X.size(0)
                counter += X.size(0)

                epoch_loss = runningloss / counter
                epoch_loss_ef = runningloss_ef / counter
                epoch_loss_esv = runningloss_esv / counter
                epoch_loss_edv = runningloss_edv / counter
                

                # str(i, runningloss, epoch_loss,  str(((summer_squared) / counter - (summer / counter)**2).item()))

                pbar.set_postfix_str("{:.2f}, {:.2f}, {:.2f}, {:.2f}, ({:.2f}) / {:.2f}".format(epoch_loss, epoch_loss_ef, epoch_loss_esv, epoch_loss_edv, loss.item(), summer_squared / counter - (summer / counter) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
        yhat_esv = np.concatenate(yhat_esv)
        yhat_edv = np.concatenate(yhat_edv)
    y = np.concatenate(y)
    y_esv = np.concatenate(y_esv)
    y_edv = np.concatenate(y_edv)

    return epoch_loss, epoch_loss_ef, epoch_loss_esv, epoch_loss_edv, yhat, yhat_esv, yhat_edv, y, y_esv, y_edv


# -

# +
def run(num_epochs=45,
        modelname="r3d_18",
        tasks=["EF", "ESV","EDV"],
        frames=16,
        period=4,
        pretrained=True,
        output=None,
        device=None,
        n_train_patients=None,
        seed=0,
        num_workers=5,
        batch_size=20,
        lr_step_period=None,
        run_test=False,
        run_extra_tests=False):

    ### Seed RNGs ###
    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(modelname, frames, period, "pretrained" if pretrained else "random"))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    if "unet3d" in modelname.split('_'):
        model = UNet3D_ef_esv_edv()
#         model = UNet3D_ef(in_channels=3, out_channels=1)
#         model.fc[2].bias.data[0] = 55.6
#         print("unet3d_separate")
#         model = UNet3D_ef_separate(in_channels=3, out_channels=1)
    else:
#         pretrain_path = './../s3d/RGB_imagenet.pkl'
#         model = model_s3d(pretrain_path,1,0.7)
        model = gen_r2p1d(**{'model_depth':50, 'pretrain' : './../3D-ResNets-PyTorch/pretrain_model/r2p1d50_KM_200ep.pth', 'funetune_size':1,'n_input_channels':3})
#         model= resnet50(**{'pretrained': False,'in_channels': 3,'num_classes': 1,'temporal_conv_layer': 1})

#             model = torchvision.models.video.__dict__[modelname[:-7]](pretrained=pretrained)

#             model.fc = torch.nn.Linear(model.fc.in_features, 1)
#             model.fc.bias.data[0] = 55.6

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)


# -

# image normalization
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

# Data preparation
    train_dataset = echonet.datasets.Echo(split="train", **kwargs, pad=12)
    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(split="val", **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
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
                loss, loss_ef, loss_esv, loss_edv, yhat, yhat_esv, yhat_edv, y, y_esv, y_edv = run_epoch(model, dataloaders[phase], phase, optim, device)
                
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              loss_ef,
                                                              loss_esv,
                                                              loss_edv,
                                                              sklearn.metrics.r2_score(yhat, y),
                                                              sklearn.metrics.r2_score(yhat_esv, y),                                                           
                                                              sklearn.metrics.r2_score(yhat_edv, y),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                              batch_size,
                                                              optim.param_groups[0]['lr']
                                                              ))
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
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                loss, loss_ef, loss_esv, loss_edv, yhat, yhat_esv, yhat_edv, y, y_esv, y_edv = run_epoch(model, dataloader, split, None, device)
                f.write("{} (one crop) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one crop) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one crop) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                
                f.write("{} (one crop) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y_esv, yhat_esv, sklearn.metrics.r2_score)))
                f.write("{} (one crop) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y_esv, yhat_esv, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one crop) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y_esv, yhat_esv, sklearn.metrics.mean_squared_error)))))
                
                f.write("{} (one crop) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y_edv, yhat_edv, sklearn.metrics.r2_score)))
                f.write("{} (one crop) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y_edv, yhat_edv, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one crop) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y_edv, yhat_edv, sklearn.metrics.mean_squared_error)))))
                f.flush()

                ds = echonet.datasets.Echo(split=split, **kwargs, crops="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                yhat = []
                yhat_esv = []
                yhat_edv = []
                y = []
                y_esv = []
                y_edv = []
                divide= 5
                for d in range(divide):
                    loss, loss_ef, loss_esv, loss_edv, yhat1, yhat_esv1, yhat_edv1, y1, y_esv1, y_edv1 = run_epoch(model, dataloader, split, None, device, save_all=True, blocks=50, flag = d, divide = divide)
                    yhat.append(yhat1)
                    y.append(y1)
                    yhat_esv.append(yhat_esv1)
                    y_esv.append(y1)
                    yhat_edv.append(yhat_edv1)
                    y_edv.append(y1)
#                 loss, yhat1, y1 = run_epoch(model, dataloader, split, None, device, save_all=True, blocks=50, flag = 1)
#                 yhat.append(yhat1)
#                 y.append(y1)
                yhat = np.concatenate(yhat)
                y = np.concatenate(y)
                yhat_esv = np.concatenate(yhat_esv)
                y_esv = np.concatenate(y_esv)
                yhat_edv = np.concatenate(yhat_edv)
                y_edv = np.concatenate(y_edv)
                f.write("{} (all crops) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all crops) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all crops) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                
                f.write("{} (all crops) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y_esv, np.array(list(map(lambda x: x.mean(), yhat_esv))), sklearn.metrics.r2_score)))
                f.write("{} (all crops) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y_esv, np.array(list(map(lambda x: x.mean(), yhat_esv))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all crops) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y_esv, np.array(list(map(lambda x: x.mean(), yhat_esv))), sklearn.metrics.mean_squared_error)))))
                
                f.write("{} (all crops) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y_edv, np.array(list(map(lambda x: x.mean(), yhat_edv))), sklearn.metrics.r2_score)))
                f.write("{} (all crops) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y_edv, np.array(list(map(lambda x: x.mean(), yhat_edv))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all crops) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y_edv, np.array(list(map(lambda x: x.mean(), yhat_edv))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f},{:.4f},{:.4f}\n".format(filename, i, p, yhat_esv[i], yhat_edv[i]))
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
# run(modelname="unet3d_ef_noNew",
#         frames=128,
#         period=1,
#         pretrained=False,
#         batch_size=2,
#         run_test=True,
#         num_epochs = 50)

run(modelname="unet3d_mt3",
        frames=32,
        period=2,
        pretrained=True,
        batch_size=16,
        lr_step_period=15,
        run_test=True,
        num_epochs = 50)

# +
# run(modelname="r2plus1d_18_author",
#         frames=32,
#         period=2,
#         pretrained=True,
#         batch_size=8,
#         run_test=True,
#         num_epochs = 45)
