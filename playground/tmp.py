# -*- coding: utf-8 -*-

import pandas as pd
from math import exp 
import numpy as np
from matplotlib import pyplot as plt
import echonet
from PIL import Image
from pyflow import pyflow

def rgb2gray(rgb):
    return np.dot(np.transpose(rgb,(1,2,0)), [0.2989, 0.5870, 0.1140])

def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def calcflow(video):
    # input size (batch, chn, time, w, h)
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    # flowout = torch.FloatTensor(2,(video.shape[1]-1),video.shape[2],video.shape[3])
    flowout = np.zeros((2,(video.shape[1]-1),video.shape[2],video.shape[3]))
    for i in range(video.shape[1] - 1):
        im1 = rgb2gray(video[:,i,:,:])
        im2 = rgb2gray(video[:,i+1,:,:])
        u, v, im2W = pyflow.coarse2fine_flow(im1[...,np.newaxis], im2[...,np.newaxis], alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
        flowout[0,i,:,:] = normalize(u)
        flowout[1,i,:,:] = normalize(v)
    return flowout

video = echonet.utils.loadvideo('/home/zdadadaz/Desktop/course/medical/data/EchoNet-Dynamic/Videos/0X1A0A263B22CCD966.avi')

# plt.imshow(np.uint8(np.transpose(video[:,0,:,:],(1,2,0))))
# plt.savefig('test.png')

# plt.imshow(np.uint8(video[0,0,:,:]))
# plt.savefig('test1.png')

# mat = np.uint8(np.transpose(video[:,0,:,:],(1,2,0)))
# mat1 = np.uint8(video[0,0,:,:])
# mat11 = rgb2gray(video[:,0,:,:])

# plt.imshow(mat11)
# plt.savefig('test11.png')

aa = calcflow(video[:,:50,:,:])
qq = (aa)
plt.imshow(np.uint8(qq[0,0,...]*255))
plt.savefig('test.png')
