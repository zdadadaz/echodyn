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
from echonet.models.unet3d import UNet3D, UNet3D_multi
# from echonet.models.deeplabv3 import DeepLabV3_multi_main
from echonet.models.r2plus1D import R2Plus1D_unet_multi

import sklearn.metrics
from echonet.datasets.echo import Echo
from echonet.datasets.echo_3d import Echo3D

def run_epoch(model, dataloader, phase, optim, device, blocks=None, flag=3):

    total = 0.
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(phase == 'train')

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []
    
    ef_criteria = torch.nn.MSELoss()
    yhat_ef = []
    y_ef = []
    runningloss_ef = 0
    count = 0
    half_len = int(len(dataloader)/2)
    with torch.set_grad_enabled(phase == 'train'):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (i, (X, (large_trace, small_trace, ef), fid)) in enumerate(dataloader):
                if flag == 0 and i > half_len:
                    pbar.update()
                    continue
                if flag == 1 and i <= half_len:
                    pbar.update()
                    continue
                
                ef = ef.to(device)
                y_ef.append(ef.cpu().numpy())
                X = X.to(device)
                average = (len(X.shape) == 6)
                if average:
                    batch, n_crops, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)
                    
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                batchidx = torch.tensor(range(X.shape[0])).to(device)
                fidlg = fid[1].to(device)
                fidsm = fid[0].to(device)
                large_trace = large_trace.to(device)
                small_trace = small_trace.to(device)
                if blocks is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + blocks), ...]) for j in range(0, X.shape[0], blocks)])

                # large frame
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(outputs[batchidx, 0, fidlg, :, :], large_trace, reduction="sum")
                large_inter += np.logical_and(outputs[batchidx, 0, fidlg, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(outputs[batchidx, 0, fidlg, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(outputs[batchidx, 0, fidlg, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum(2).sum(1))
                large_union_list.extend(np.logical_or(outputs[batchidx, 0, fidlg, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum(2).sum(1))

                # small frame
                outputs = model(X)
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(outputs[batchidx, 0, fidsm, :, :], small_trace, reduction="sum")
                small_inter += np.logical_and(outputs[batchidx, 0, fidsm, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(outputs[batchidx, 0, fidsm, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(outputs[batchidx, 0, fidsm, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum(2).sum(1))
                small_union_list.extend(np.logical_or(outputs[batchidx, 0, fidsm, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum(2).sum(1))

                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()
                
                # yhat_ef.append(ef_small.view(-1).to("cpu").detach().numpy())
                yhat_ef.append(ef.cpu().numpy())
                
                # loss_ef_large = ef_criteria(ef_large.view(-1)/100, edv/100)
                
                loss_seg = (loss_large + loss_small) / 2 
                # loss_ef = (loss_ef_large + loss_ef_small)/2
                loss = loss_seg 
                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss_seg.item()
                n += large_trace.size(0)
                # runningloss_ef += loss_ef.item() * large_frame.size(0)

                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)
                
                total_seg = total / n / 112 / 112
                
                epoch_loss = total_seg
                
                pbar.set_postfix_str("tot: {:.4f}, ef: {:.4f}, seg: {:4f}".format(epoch_loss, runningloss_ef/n, total_seg))
                
                # pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, 
                #                                                                               loss.item() / large_trace.size(0) / 112 / 112, 
                #                                                                               -p * math.log(p) - (1 - p) * math.log(1 - p), 
                #                                                                               (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 
                #                                                                               2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    # y_esv = np.concatenate(y_esv)
    # y_edv = np.concatenate(y_edv)
    # yhat_esv = np.concatenate(yhat_esv)
    # yhat_edv = np.concatenate(yhat_edv)
    if not save_all:
        yhat_ef = np.concatenate(yhat_ef)
    y_ef = np.concatenate(y_ef)
    
    return (epoch_loss,
            total_seg,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            runningloss_ef/n,
            yhat_ef,
            y_ef
            )


def run(num_epochs=50,
        modelname="unet",
        pretrained=False,
        output=None,
        device=None,
        frames=32,
        period=2,
        n_train_patients=None,
        num_workers=5,
        batch_size=8,
        seed=0,
        lr_step_period=None,
        save_segmentation=False):

    ### Seed RNGs ###
    np.random.seed(seed)
    torch.manual_seed(seed)

    tasks = [ "LargeTrace", "SmallTrace", "EF"]

    if output is None:
#         output = os.path.join("output", "segmentation", "{}_{}".format(modelname, "pretrained" if pretrained else "random"))
        output = os.path.join("output", "segmentation", "{}_{}_{}_{}".format(modelname, frames, period, "pretrained" if pretrained else "random"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    if "unet3d" in modelname.split('_'):
        model = UNet3D(in_channels=3, out_channels=1)
    else:
#         model = DeepLabV3_multi_main()
        model = R2Plus1D_unet_multi(layer_sizes=[2, 2, 2, 2])
        # model = torchvision.models.segmentation.__dict__[modelname](pretrained=pretrained, aux_loss=False)
        
    # print(model)
    # model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    optim = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    mean, std = echonet.utils.get_mean_and_std(Echo(split="train"))
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    train_dataset = Echo3D(split="train", **kwargs)
    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        Echo3D(split="val", **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # input 3x112x112
    
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
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

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)

                loss, seg_loss, large_inter, large_union, small_inter, small_union, ef_loss, yhat_ef, y_ef = run_epoch(model, dataloaders[phase], phase, optim, device)
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    overall_dice,
                                                                    large_dice,
                                                                    small_dice,
                                                                    time.time() - start_time,
                                                                    large_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                                    batch_size,
                                                                    ef_loss,
                                                                    sklearn.metrics.r2_score(yhat_ef, y_ef)))
                f.flush()
            scheduler.step()

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'ef_loss':ef_loss,
                'ef_r2': sklearn.metrics.r2_score(yhat_ef, y_ef),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        for split in ["val", "test"]: 
            print(split)
            dataset = Echo3D(split=split, **kwargs)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
            loss, seg_loss, large_inter, large_union, small_inter, small_union, ef_loss, yhat_ef, y_ef = run_epoch(model, dataloader, split, None, device)

            overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
            large_dice = 2 * large_inter / (large_union + large_inter)
            small_dice = 2 * small_inter / (small_union + small_inter)
            with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                g.write("Filename, Overall, Large, Small\n")
                for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                    g.write("{},{},{},{}\n".format(filename, overall, large, small))

            f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), echonet.utils.dice_similarity_coefficient)))
            f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(large_inter, large_union, echonet.utils.dice_similarity_coefficient)))
            f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(small_inter, small_union, echonet.utils.dice_similarity_coefficient)))
            f.flush()
            
            # with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
            #     for i, (filename, pred) in enumerate(zip(dataset.fnames, yhat_esv)):
            #         g.write("{},{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(filename, i, pred,y_esv[i], yhat_edv[i], y_edv[i]))
            
            echonet.utils.latexify()
            fig = plt.figure(figsize=(4, 4))
            plt.scatter(small_dice, large_dice, color="k", edgecolor=None, s=1)
            plt.plot([0, 1], [0, 1], color="k", linewidth=1)
            plt.axis([0, 1, 0, 1])
            plt.xlabel("Systolic DSC")
            plt.ylabel("Diastolic DSC")
            plt.tight_layout()
            plt.savefig(os.path.join(output, "{}_dice.pdf".format(split)))
            plt.savefig(os.path.join(output, "{}_dice.png".format(split)))
            plt.close(fig)

    def collate_fn(x):
        x, f = zip(*x)
        i = list(map(lambda t: t.shape[1], x))
        x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
        return x, f, i
    
    # Save labels of all videos (labels folder)
    dataloader = torch.utils.data.DataLoader(Echo(split="all", target_type=["Filename"], length=None, period=1, mean=mean, std=std),
                                             batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), collate_fn=collate_fn)
    if save_segmentation and not all([os.path.isfile(os.path.join(output, "labels", os.path.splitext(f)[0] + ".npy")) for f in dataloader.dataset.fnames]):
        # Save segmentations for all frames
        # Only run if missing files

        pathlib.Path(os.path.join(output, "labels")).mkdir(parents=True, exist_ok=True)
        block = 1024
        model.eval()

        with torch.no_grad():
            with open(os.path.join(output, "{}_EF_predictions.csv".format(modelname)), "w") as gp:
                for (x, f, i) in tqdm.tqdm(dataloader):
                    x = x.to(device)
                    tmp = [model(x[i:(i + block), :, :, :]) for i in range(0, x.shape[0], block)]
                    y = []
                    ef = []
                    for t in tmp:
                        y.append(t[0].detach().cpu().numpy())
                        ef.append(t[1].detach().cpu().numpy())
                    y = np.concatenate(y)
                    ef = np.concatenate(ef)
                    ef = ef.reshape(-1)
                    # y = np.concatenate([model(x[i:(i + block), :, :, :]) for i in range(0, x.shape[0], block)]).astype(np.float16)
                    start = 0
                    for (filename, offset) in zip(f, i):
                        count = 0
                        np.save(os.path.join(output, "labels", os.path.splitext(filename)[0]), y[start:(start + offset), 0, :, :])
                        for e in ef[start:(start + offset)]:
                            gp.write("{},{},{:.4f}\n".format(filename, str(count), e))
                            count+=1
                        start += offset
        
    # Save size for all videos (videos folder)
    dataloader = torch.utils.data.DataLoader(Echo(split="all", target_type=["Filename", "LargeIndex", "SmallIndex"], length=None, period=1, segmentation=os.path.join(output, "labels")),
                                             batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=False)
    if save_segmentation and not all([os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames]):
        pathlib.Path(os.path.join(output, "videos")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(output, "size")).mkdir(parents=True, exist_ok=True)
        echonet.utils.latexify()
        with open(os.path.join(output, "size.csv"), "w") as g:
            g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
            for (x, (filename, large_index, small_index)) in tqdm.tqdm(dataloader):
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
                    trim_min = sorted(size)[round(len(size) ** 0.05)]
                    trim_max = sorted(size)[round(len(size) ** 0.95)]
                    trim_range = trim_max - trim_min
                    peaks = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
                    for (x, y) in enumerate(size):
                        g.write("{},{},{},{},{},{}\n".format(filename[0], x, y, 1 if x == large_index[i] else 0, 1 if x == small_index[i] else 0, 1 if x in peaks else 0))
                    # fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                    # plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                    # ylim = plt.ylim()
                    # for p in peaks:
                    #     plt.plot(np.array([p, p]) / 50, ylim, linewidth=1)
                    # plt.ylim(ylim)
                    # plt.title(os.path.splitext(filename[i])[0])
                    # plt.xlabel("Seconds")
                    # plt.ylabel("Size (pixels)")
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(output, "size", os.path.splitext(filename[i])[0] + ".pdf"))
                    # plt.close(fig)
                    # size -= size.min()
                    # size = size / size.max()
                    # size = 1 - size
                    # for (x, y) in enumerate(size):
                    #     img[:, :, int(round(115 + 100 * y)), int(round(x / len(size) * 200 + 10))] = 255.
                    #     interval = np.array([-3, -2, -1, 0, 1, 2, 3])
                    #     for a in interval:
                    #         for b in interval:
                    #             img[:, x, a + int(round(115 + 100 * y)), b + int(round(x / len(size) * 200 + 10))] = 255.

                    #             if x == large_index[i]:
                    #                 img[0, :, a + int(round(115 + 100 * y)), b + int(round(x / len(size) * 200 + 10))] = 255.
                    #             if x == small_index[i]:
                    #                 img[1, :, a + int(round(115 + 100 * y)), b + int(round(x / len(size) * 200 + 10))] = 255.
                    #     if x in peaks:
                    #         img[:, :, 200:225, b + int(round(x / len(size) * 200 + 10))] = 255.
                    # echonet.utils.savevideo(os.path.join(output, "videos", filename[i]), img.astype(np.uint8), 50)


echonet.config.DATA_DIR = '../../data/EchoNet-Dynamic'
run(num_epochs=50,
        modelname="r2plus1d_seg",
        frames=32,
        period=2,
        pretrained=False,
        batch_size=6,
        save_segmentation=False)

# run(num_epochs=50,
#         modelname="unet3D_seg",
#         frames=112,
#         period=1,
#         pretrained=False,
#         batch_size=2,
#         save_segmentation=False)
