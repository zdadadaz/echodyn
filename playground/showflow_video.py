# -*- coding: utf-8 -*-

import os
import numpy as np
from pyflow import pyflow
import cv2
import matplotlib.pyplot as plt

video_path = '/home/zdadadaz/Desktop/course/medical/data/EchoNet-Dynamic/Videos/0X1A0A263B22CCD966.avi'

cap = cv2.VideoCapture(video_path)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

count = 1
while(1):
    
    ret, frame2 = cap.read()
    if count%2 == 1:
        count += 1
        continue
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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
    
    
    
    hsv[...,0] = ang*180/np.pi/2
    hsv[mag < 0.1,0] = 90
    mag[mag < 0.1] = 0
    hist, bin_edges = np.histogram(mag)
    plt.plot(hist)
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    
    # hsv[...,0] = cv2.normalize(flow[...,0],None,0,255,cv2.NORM_MINMAX)
    # hsv[...,2] = cv2.normalize(flow[...,1],None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)

    
    cv2.imwrite('./output/test/org_{}.png'.format(str(count)),frame2)
    cv2.imwrite('./output/test/opticalhsv_m_{}.png'.format(str(count-1)),rgb)
    prvs = next
    count+= 1

cap.release()
cv2.destroyAllWindows()