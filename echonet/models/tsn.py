import torch
import torch.nn as nn
import torchvision
import numpy as np
from echonet.models.resnet3d import resnet50 

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        self.shape = input.size()
        if self.consensus_type == 'avg':
            output = input.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input
        else:
            output = None
        return output


class TSN(nn.Module):
    def __init__(self, batch=1, pretrained = True):
        super(TSN, self).__init__()
        self.pretrained = pretrained
        self._prepare_model()
        self.consensus = ConsensusModule('avg')
        self.batch = batch
    def forward(self, X, flow):
        # batch*n_crops, c, f, h, w
        flow = self.model_f(flow)
        rgb = self.model(X)
        flow_out = self.consensus(flow.view(self.batch,-1))
        rgb_out = self.consensus(rgb.view(self.batch,-1))
        out = torch.div(torch.add(flow_out, rgb_out),2)
        return out  # (batch,1)
    
    def _prepare_model(self):
        self.model_f = torchvision.models.video.__dict__['r2plus1d_18'](pretrained=self.pretrained)
        self.model_f.stem[0] = nn.Conv3d(2, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.model_f.fc = torch.nn.Linear(self.model_f.fc.in_features, 1)
        self.model_f.fc.bias.data[0] = 55.6
        
        self.model = torchvision.models.video.__dict__['r2plus1d_18'](pretrained=self.pretrained)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        self.model.fc.bias.data[0] = 55.6
#         print(self.model_f.stem[0].weight.data.shape, self.model.stem[0].weight.data.shape)
        # assign init
        self.model_f.stem[0].weight.data = self.model.stem[0].weight.data[:,:2,...]


class TSN_resnet(nn.Module):
    def __init__(self, batch=1, pretrained = True):
        super(TSN, self).__init__()
        self.pretrained = pretrained
        self._prepare_model()
        self.consensus = ConsensusModule('avg')
        self.batch = batch
    def forward(self, X, flow):
        # batch*n_crops, c, f, h, w
        flow = self.model_f(flow)
        rgb = self.model(X)
        flow_out = self.consensus(flow.view(self.batch,-1))
        rgb_out = self.consensus(rgb.view(self.batch,-1))
        out = torch.div(torch.add(flow_out, rgb_out),2)
        return out  # (batch,1)
    
    def _prepare_model(self):
        self.model_f = resnet50(**{'pretrained': False,'in_channels': 2,'num_classes': 1,'temporal_conv_layer': 1})
        self.model = resnet50(**{'pretrained': False,'in_channels': 3,'num_classes': 1,'temporal_conv_layer': 1})



# +
model = TSN_resnet(batch = 3)
# model = TSN(batch = 3)
# batch,ch,segment*len,w,h
X = torch.rand(3*2,3,32,112,112)
flow = torch.rand(3*2,2,32,112,112)
out = model(X,flow)
print(out)

# x = torch.tensor([[[1, 2, 3], [4, 5, 6]],[[1, 2, 3], [4, 5, 6]]])
# print(x.size())
# print(x)
# y = x.view(4,3)
# print(y.size())
# print(y)
# print(y.view(2,2,3))
