# -*- coding: utf-8 -*-

import os
import numpy as np
from pyflow import pyflow
import cv2


flow_path = '/home/zdadadaz/Desktop/course/medical/data/EchoNet-Dynamic/flow_2/0X1A0A263B22CCD966.avi/'

alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 1  # 0 or d

hsv = np.zeros((112,112,3))
hsv[...,1] = 255

videolist = [32,36,38,40,42,44,46,48,50,52,54,56,58,60,72]

for i in range(len(videolist)):
    frame_idx = 'frame'+ str(videolist[i]).zfill(6)
    u = cv2.imread(os.path.join(flow_path+"/u/", frame_idx +'.jpg'),cv2.IMREAD_GRAYSCALE)/255.
    v = cv2.imread(os.path.join(flow_path+"/v/", frame_idx +'.jpg'),cv2.IMREAD_GRAYSCALE)/255.
    
    mag, ang = cv2.cartToPolar(u, v)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # hsv[...,0] = np.uint8(u*255)
    # hsv[...,2] = np.uint8(v*255)
    rgb = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)

    cv2.imwrite('./output/opticalhsv_{}.png'.format(str(i)),rgb)
    