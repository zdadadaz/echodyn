# -*- coding: utf-8 -*-

import os
import numpy as np
from pyflow import pyflow
import cv2

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
    hsv[...,0] = ang*180/np.pi/2
    print(hsv[...,0].max(), hsv[...,0].min())
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    
    cv2.imwrite('./output/test/org_{}.png'.format(str(count)),frame2)
    cv2.imwrite('./output/test/opticalhsv_{}.png'.format(str(count-1)),rgb)
    prvs = next
    count+= 1

cap.release()
cv2.destroyAllWindows()