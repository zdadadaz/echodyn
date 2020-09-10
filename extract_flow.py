import os
import numpy as np
import cv2
from glob import glob
from multiprocessing import Pool
import pandas as pd
import echonet
import pathlib
import time

def rgb2gray(rgb):
    return np.dot(np.transpose(rgb,(1,2,0)), [0.2989, 0.5870, 0.1140])


def loadvideo(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError()
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # empty numpy array of appropriate length, fill in when possible from front
    v = np.zeros((frame_count, frame_width, frame_height, 3), np.float32)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    v = v.transpose((3, 0, 1, 2))

    return v


def normalize(x, minV, maxV):
    # 112*0.1 = 11 pixel movement
#     minV = -0.1
#     maxV = 0.1
    out = (x - minV)/ (maxV - minV)
    out[out > 1] = 1
    out[out<0] = 0
    return out


def cvflow(prvs,next):
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prvs, next, None)
    u = flow[...,0]
    v = flow[...,1]
    return normalize(u, -20.0, 20.0), normalize(v, -20.0, 20.0) 


def calcflow(video, path, videolist,filename,period, save=False):
    # input size (chn, time, w, h)
    # Flow Options:
    flowout = np.zeros((2,(video.shape[1]-1),video.shape[2],video.shape[3]),dtype=video.dtype)
    for i in range(video.shape[1] - period):
        frame_idx = 'frame'+ str(videolist[i]).zfill(6)
        frame_idx_n = 'frame'+ str(videolist[i+period]).zfill(6)
        if os.path.exists(os.path.join(path+"/u/", frame_idx +'_'+frame_idx_n+'.jpg')):
            pass
        else:
            im1 = np.float32(rgb2gray(video[:,i,:,:]))
            im2 = np.float32(rgb2gray(video[:,i+period,:,:]))
            flowout[0,i,:,:], flowout[1,i,:,:] = cvflow(im1,im2)     
            if save:   
                cv2.imwrite(os.path.join(path+"/u/", frame_idx +'_'+frame_idx_n+'.jpg'),np.uint8(flowout[0,i,:,:]*255),[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(os.path.join(path+"/v/", frame_idx +'_'+frame_idx_n+'.jpg'),np.uint8(flowout[1,i,:,:]*255),[int(cv2.IMWRITE_JPEG_QUALITY), 90])



def gen_video_path(split):
    base = '../../data/EchoNet-Dynamic/'
    df = pd.read_csv(base+'FileList.csv')
    df = df[df['Split']==split.upper()]
    files = list(df['FileName'])
    out = [base + 'Videos/' + i for i in files]
    return out


def extract_flow(args):
    t = time.time()
    video_path, mean, std, period = args
    flowPath = '../../data/EchoNet-Dynamic/tvflow_xy_2/'
    fnames = video_path.split('/')[-1]
    print(video_path)
    video = loadvideo(video_path)
    frame_count = video.shape[1]
    videolist = [i for i in range(frame_count)]
    if isinstance(mean, int) or isinstance(mean, float):
        video = (video - mean) / std
    else:
        video = (video - mean.reshape(3, 1, 1, 1)) / std.reshape(3, 1, 1, 1)
    
    flow_filename = os.path.join(flowPath,fnames)
    pathlib.Path(os.path.join(flow_filename,'u')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(flow_filename,'v')).mkdir(parents=True, exist_ok=True)
    calcflow(video, flow_filename,videolist,fnames,period, save=True)
    # do stuff
    elapsed = time.time() - t
    print('complete:' + flow_filename +' cost time : '+ str(elapsed))
    return


if __name__ =='__main__':
    echonet.config.DATA_DIR = '../../data/EchoNet-Dynamic'
    period = 2
    pool = Pool(20)   # multi-processing
    
    splits = ['train', 'test','val']
    for split in splits:
#     split = 'train'
        video_paths = gen_video_path(split=split)
        mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split=split))
        means = [mean for i in range(len(video_paths))]
        stds = [std for i in range(len(video_paths))]
        periods = [period for i in range(len(video_paths))]
        pool.map(extract_flow, zip(video_paths, means, stds, periods))
