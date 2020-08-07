import echonet
import pathlib
import torch.utils.data
import os
import numpy as np
import collections
import skimage.draw
from pyflow import pyflow
import cv2

def rgb2gray(rgb):
    return np.dot(np.transpose(rgb,(1,2,0)), [0.2989, 0.5870, 0.1140])


def normalize(x, minV, maxV):
    # 112*0.1 = 11 pixel movement
#     minV = -0.1
#     maxV = 0.1
    minV = max(minV,x.min())
    maxV = min(maxV, x.max())
    out = (x - minV)/ (maxV - minV)
    out[out > 1] = 1
    out[out<0] = 0
    return out

def cvflow(prvs,next):
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                                        #scale,level, size,iteration,poly_n,ploy_sigma,flag
#     flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.8, 3, 7, 10, 5, 1.2, 0)
#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    u = flow[...,0]
    v = flow[...,1]
    return normalize(u,-0.1,0.1), normalize(v,-0.1,0.1)

def calcflow(video, path, videolist,filename, save=False):
    # input size (chn, time, w, h)
    # Flow Options:
    alpha = 0.197
    ratio = 0.8
    minWidth = 20
    nOuterFPIterations = 77
    nInnerFPIterations = 10
    nSORIterations = 10
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    # flowout = torch.FloatTensor(2,(video.shape[1]-1),video.shape[2],video.shape[3])
#     flowout = np.zeros((2,(video.shape[1]-1),video.shape[2],video.shape[3]))
    flowout = np.zeros((3,(video.shape[1]-1),video.shape[2],video.shape[3]))
    for i in range(video.shape[1] - 1):
#         print(filename)
        frame_idx = 'frame'+ str(videolist[i]).zfill(6)
        frame_idx_n = 'frame'+ str(videolist[i+1]).zfill(6)
        if os.path.exists(os.path.join(path+"/u/", frame_idx +'_'+frame_idx_n+'.jpg')):
            u = cv2.imread(os.path.join(path+"/u/", frame_idx +'_'+frame_idx_n+'.jpg'),cv2.IMREAD_GRAYSCALE)/255.
            v = cv2.imread(os.path.join(path+"/v/", frame_idx +'_'+frame_idx_n+'.jpg'),cv2.IMREAD_GRAYSCALE)/255.
            flowout[0,i,:,:] = u
            flowout[1,i,:,:] = v
        else:
            im1 = rgb2gray(video[:,i,:,:])
            im2 = rgb2gray(video[:,i+1,:,:])
#             u, v, im2W = pyflow.coarse2fine_flow(im1[...,np.newaxis], im2[...,np.newaxis], alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
#             nSORIterations, colType)
#             flowout[0,i,:,:] = normalize(u, -10.0, 10.0)
#             flowout[1,i,:,:] = normalize(v, -10.0, 10.0)
            flowout[0,i,:,:], flowout[1,i,:,:] = cvflow(im1,im2)
            
            if save:   
                cv2.imwrite(os.path.join(path+"/u/", frame_idx +'_'+frame_idx_n+'.jpg'),np.uint8(flowout[0,i,:,:]*255),[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(os.path.join(path+"/v/", frame_idx +'_'+frame_idx_n+'.jpg'),np.uint8(flowout[1,i,:,:]*255),[int(cv2.IMWRITE_JPEG_QUALITY), 90])
        flowout[2,i,:,:] = rgb2gray(video[:,i,:,:])
    return flowout

class Echo3Df(torch.utils.data.Dataset):
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=4,
                 max_length=250,
                 crops=1,
                 pad=None,
                 noise=None,
                 segmentation=None,
                 target_transform=None,
                 external_test_location=None):
        """length = None means take all possible"""

        if root is None:
            root = echonet.config.DATA_DIR
            
        self.folder = pathlib.Path(root)
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.crops = crops
        self.pad = pad
        self.noise = noise
        self.segmentation = segmentation
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        elif split == "clinical_test":
            self.fnames = sorted(os.listdir(self.folder / "ProcessedStrainStudyA4c"))
        else:
            with open(self.folder / "FileList.csv") as f:
                self.header = f.readline().strip().split(",")
                filenameIndex = self.header.index("FileName")
                splitIndex = self.header.index("Split")

                for (i, line) in enumerate(f):
                    lineSplit = line.strip().split(',')

                    fileName = lineSplit[filenameIndex]
                    fileMode = lineSplit[splitIndex].lower()

                    if (split == "all" or split == fileMode) and os.path.exists(self.folder / "Videos" / fileName):
                        self.fnames.append(fileName)
                        self.outcome.append(lineSplit)

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(self.folder / "VolumeTracings.csv") as f:
                header = f.readline().strip().split(",")

                for (i, line) in enumerate(f):
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            keep = [len(self.frames[os.path.splitext(f)[0]]) >= 2 and f != "0X4F55DC7F6080587E.avi" for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        if self.split == "external_test":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.folder, "Videos", self.fnames[index])
        video = echonet.utils.loadvideo(video)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        assert(type(self.mean) == type(self.std))
        if isinstance(self.mean, int) or isinstance(self.mean, float):
            video = (video - self.mean) / self.std
        else:
            video = (video - self.mean.reshape(3, 1, 1, 1)) / self.std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        if self.length is None:
            length = f // self.period
        else:
            length = self.length

        length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        # small
        first_frameid = int(self.frames[os.path.splitext(self.fnames[index])[0]][0])
        # large
        last_frameid = int(self.frames[os.path.splitext(self.fnames[index])[0]][-1])
        if self.crops == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            fromEnd = max(np.random.choice(f - (length - 1) * self.period, self.crops),1)
            # tmp  = max(1,last_frameid - (length - 1) * self.period)
            firstframe = min(first_frameid, last_frameid)
            tmp = min(firstframe,fromEnd)
            try:
                if firstframe == tmp:
                    start = 0
                else:
                    start = np.random.choice(tmp, self.crops)
            except:
                print("tmp is <1 in random choice")
                print(tmp,firstframe, fromEnd)
        target = []
        
        for t in self.target_type:
            key = os.path.splitext(self.fnames[index])[0]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t == "LargeTrace" or t == "SmallTrace":
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            elif t == "EF":
                if self.split == "clinical_test" or self.split == "external_test":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if self.segmentation is not None:
            seg = np.load(os.path.join(self.segmentation, os.path.splitext(self.fnames[index])[0] + ".npy"))
            video[2, :seg.shape[0], :, :] = seg

        # print(self.fnames[index])

        # Select random crops
        if self.crops==1:
            videolist =list(start + self.period * np.arange(length))
            videolist.append(first_frameid)
            videolist.append(last_frameid)
            videolist = sorted(list(set(videolist)))
            lendiff = abs(len(videolist)-length)
            def resizelist(vlist,first_frameid,last_frameid, lendiff):
               for v in range(len(vlist)):
                   if vlist[v] != first_frameid and lendiff > 0:
                       vlist.pop(v)
                       lendiff -= 1
                   if vlist[len(vlist)-1-v] != last_frameid and lendiff > 0:
                       vlist.pop(len(vlist)-1-v)
                       lendiff -= 1
                   if lendiff==0:
                       return vlist

            
            videolist = resizelist(videolist, min(first_frameid, last_frameid), max(first_frameid, last_frameid), lendiff)
            # print(first_frameid, last_frameid, lendiff)
            # print(len(videolist))
            # print(videolist)
            try:
                frameid = [list(videolist).index(first_frameid), list(videolist).index(last_frameid)]
            except:
                print("videolist")
                print(videolist)
                print("first_frameid", first_frameid)
                print("last_frameid", last_frameid)
                
            # print("frameid: ", frameid)
            videolist = np.array(videolist)
            video = video[:, videolist, :, :]
            
        else:
            frameid = [first_frameid, last_frameid] 
            videolist =tuple([s + self.period * np.arange(length) for s in start])
            video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
            
        
        # if self.crops == 1:
        #     video = video[0]
        # else:
        if self.crops != 1:
            video = np.stack(video)
        # print(video.shape)
        flowPath = os.path.join(self.folder,"flow_xy_" + str(self.period))
        for t in self.target_type:
            if t == 'flow':
                # print(self.fnames[index])
                flow_filename = os.path.join(flowPath,self.fnames[index])
                pathlib.Path(os.path.join(flow_filename,'u')).mkdir(parents=True, exist_ok=True)
                pathlib.Path(os.path.join(flow_filename,'v')).mkdir(parents=True, exist_ok=True)    
                if self.crops ==1:
                    flowout = calcflow(video, flow_filename,videolist,self.fnames[index],  save=True)
                    target.append(flowout)
                else:
                    flow_buf = []
                    for c in range(video.shape[0]):
                        flowout = calcflow(video[c], flow_filename,videolist[c],self.fnames[index],  save=True)
                        flow_buf.append(flowout)
                    flow_buf = tuple(flow_buf)
                    flow_buf = np.stack(tuple(flow_buf))
                    target.append(flow_buf)
                    
        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
                
        if self.pad is not None:
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, target, frameid

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    return collections.defaultdict(list)
