# -*- coding: utf-8 -*-

import os
import numpy as np
# from pyflow import pyflow
import cv2
import matplotlib.pyplot as plt
import echonet
from echonet.datasets.echo import Echo

def normalize(x, minV, maxV):
    # minV = -0.04
    # maxV = 0.04
    out = (x - minV)/ (maxV - minV)
    out[out > 1] = 1
    out[out<0] = 0
    return out

def cvflow(prvs,next):
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 7, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    return (cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX), ang/np.pi/2)
    

video_path = '/Users/chienchichen/Desktop/UQ/capstone/medical/data/EchoNet-Dynamic/Videos/0X1A0A263B22CCD966.avi'
echonet.config.DATA_DIR = '../../../data/EchoNet-Dynamic'

cap = cv2.VideoCapture(video_path)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
mean, std = echonet.utils.get_mean_and_std(Echo(split="train"))
mean, std = mean[0],std[0]
prvs = (prvs-mean)/std

count = 1
while(1):
    
    ret, frame2 = cap.read()
    if count%2 == 1:
        count += 1
        continue
    if frame2 is None:
        break
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    next = (next-mean)/std
    
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 7, 3, 5, 1.2, 0)
    # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 7, 10, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # alpha = 0.012
    # ratio = 0.75
    # minWidth = 20
    # nOuterFPIterations = 7
    # nInnerFPIterations = 1
    # nSORIterations = 30
    # colType = 1 
    # u, v, im2W = pyflow.coarse2fine_flow(np.double(prvs[...,np.newaxis]), np.double(next[...,np.newaxis]), alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    #         nSORIterations, colType)
    # mag, ang = cv2.cartToPolar(u, v)
    
    # out1, out2 = (cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX), ang/np.pi/2)
    # out1, out2 = (cv2.normalize(flow[...,0],None,0,1,cv2.NORM_MINMAX), cv2.normalize(flow[...,1],None,0,1,cv2.NORM_MINMAX))
    # out1, out2 = normalize(flow[...,0], -3.0, 3.0), normalize(flow[...,1], -3.0, 3.0) 
    out1, out2 = normalize(flow[...,0], -0.1, 0.1), normalize(flow[...,1], -0.1, 0.1) 
    
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[mag < 0.1,0] = 0
    # mag[mag < 0.1] = 0
    print(flow[...,0].min(),flow[...,0].max())
    print(out1.min(),out1.max())
    print(flow[...,1].min(),flow[...,1].max())
    print(out2.min(),out2.max())
    
    hist, bin_edges = np.histogram(flow[...,0])
    plt.plot(hist)
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    
    # hsv[...,0] = cv2.normalize(flow[...,0],None,0,255,cv2.NORM_MINMAX)
    # hsv[...,2] = cv2.normalize(flow[...,1],None,0,255,cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)
    # rgb = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_GRAY2RGB)
    
    # cv2.imwrite('./output_tmp/org_{}.png'.format(str(count)),frame2)
    cv2.imwrite('./output_tmp/opticalhsv_m_{}_mag.png'.format(str(count-1)),np.uint8(out1*255))
    cv2.imwrite('./output_tmp/opticalhsv_m_{}_ang.png'.format(str(count-1)),np.uint8(out2*255))
    prvs = next
    count+= 1

cap.release()
cv2.destroyAllWindows()