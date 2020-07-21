import torch
import torch.nn as nn
import torchvision
import numpy as np
from echonet.models.resnet3d import resnet50 
# from resnet3d import resnet50 

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
#         
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


# +
class TSN_resnet(nn.Module):
    def __init__(self, batch=1, pretrained = True):
        super(TSN_resnet, self).__init__()
        self.pretrained = pretrained
        self._prepare_model()
        self.consensus = ConsensusModule('avg')
        self.batch = batch
    def forward(self, x_s, x_t):
        # batch*n_crops, c, f, h, w
        x_s = self.spatial_stream.conv1(x_s)
        x_s = self.spatial_stream.bn1(x_s)
        x_s = self.spatial_stream.relu(x_s)
        x_s = self.spatial_stream.maxpool(x_s)

        x_t = self.temporal_stream.conv1(x_t)
        x_t = self.temporal_stream.bn1(x_t)
        x_t = self.temporal_stream.relu(x_t)
        x_t = self.temporal_stream.maxpool(x_t)

        for i in range(1, 5):
            layer_spatial = self.spatial_stream.__getattr__('layer{}'.format(i))
            layer_temporal = self.temporal_stream.__getattr__('layer{}'.format(i))
            assert layer_spatial.blocks <= layer_temporal.blocks    # CHECKME: skipped for resnet 50 and 152
            for j in range(layer_spatial.blocks):
                x_s_res, x_t_res = None, None
                if 2 == j:
                    x_s_res = x_s                       # T -> S
#                     x_s = x_s * x_t                     # Multiplicative Modulation

                x_s = layer_spatial.__getattr__('sblock_{}'.format(j))(x_s, residual=x_s_res)
                x_t = layer_temporal.__getattr__('sblock_{}'.format(j))(x_t, residual=x_t_res)

            for j in range(layer_temporal.blocks-layer_spatial.blocks):
                x_t = layer_temporal.__getattr__('sblock_{}'.format(layer_spatial.blocks+j))(x_t, residual=x_t_res)

        x_s = self.spatial_stream.s_pool(x_s)
        x_s = self.spatial_stream.t_pool(x_s)

        x_t = self.temporal_stream.s_pool(x_t)
        x_t = self.temporal_stream.t_pool(x_t)

        x_s = functional.relu(self.spatial_stream.fc(x_s), inplace=True)
        x_t = functional.relu(self.temporal_stream.fc(x_t), inplace=True)

        rgb = x_s.view(x_s.size(0), x_s.size(1))
        flow = x_t.view(x_t.size(0), x_t.size(1))
        
#       TSN
        flow_out = self.consensus(flow.view(self.batch,-1))
        rgb_out = self.consensus(rgb.view(self.batch,-1))
        out = torch.div(torch.add(flow_out, rgb_out),2)
        return out  # (batch,1)

    def _prepare_model(self):
        self.temporal_stream = resnet50(**{'pretrained': False,'in_channels': 2,'num_classes': 1,'temporal_conv_layer': 1})
        self.spatial_stream = resnet50(**{'pretrained': False,'in_channels': 3,'num_classes': 1,'temporal_conv_layer': 1})
        


# +
class TSN_resnet_multiply(nn.Module):
    def __init__(self, batch=1, pretrained = True):
        super(TSN_resnet_multiply, self).__init__()
        self.pretrained = pretrained
        self._prepare_model()
        self.consensus = ConsensusModule('avg')
        self.batch = batch
    def forward(self, X, flow):
        # batch*n_crops, c, f, h, w
        x_s = self.spatial_stream.conv1(x_s)
        x_s = self.spatial_stream.bn1(x_s)
        x_s = self.spatial_stream.relu(x_s)
        x_s = self.spatial_stream.maxpool(x_s)

        x_t = self.temporal_stream.conv1(x_t)
        x_t = self.temporal_stream.bn1(x_t)
        x_t = self.temporal_stream.relu(x_t)
        x_t = self.temporal_stream.maxpool(x_t)

        for i in range(1, 5):
            layer_spatial = self.spatial_stream.__getattr__('layer{}'.format(i))
            layer_temporal = self.temporal_stream.__getattr__('layer{}'.format(i))
            assert layer_spatial.blocks <= layer_temporal.blocks    # CHECKME: skipped for resnet 50 and 152
            for j in range(layer_spatial.blocks):
                x_s_res, x_t_res = None, None
                if 2 == j:
                    x_s_res = x_s                       # T -> S
                    x_s = x_s * x_t                     # Multiplicative Modulation

                x_s = layer_spatial.__getattr__('sblock_{}'.format(j))(x_s, residual=x_s_res)
                x_t = layer_temporal.__getattr__('sblock_{}'.format(j))(x_t, residual=x_t_res)

            for j in range(layer_temporal.blocks-layer_spatial.blocks):
                x_t = layer_temporal.__getattr__('sblock_{}'.format(layer_spatial.blocks+j))(x_t, residual=x_t_res)

        x_s = self.spatial_stream.s_pool(x_s)
        x_s = self.spatial_stream.t_pool(x_s)

        x_t = self.temporal_stream.s_pool(x_t)
        x_t = self.temporal_stream.t_pool(x_t)

        x_s = functional.relu(self.spatial_stream.fc(x_s), inplace=True)
        x_t = functional.relu(self.temporal_stream.fc(x_t), inplace=True)

        rgb = x_s.view(x_s.size(0), x_s.size(1))
        flow = x_t.view(x_t.size(0), x_t.size(1))
        
#       flow = self.model_f(flow)
#       rgb = self.model(X)
#       TSN
        flow_out = self.consensus(flow.view(self.batch,-1))
        rgb_out = self.consensus(rgb.view(self.batch,-1))
        out = torch.div(torch.add(flow_out, rgb_out),2)
        return out  # (batch,1)
    
    def _prepare_model(self):
        self.temporal_stream = resnet50(**{'pretrained': False,'in_channels': 2,'num_classes': 1,'temporal_conv_layer': 1})
        self.spatial_stream = resnet50(**{'pretrained': False,'in_channels': 3,'num_classes': 1,'temporal_conv_layer': 1})

# +
# model = TSN_resnet(batch = 3)
# model = TSN(batch = 3)
# batch,ch,segment*len,w,h
# X = torch.rand(3*2,3,32,112,112)
# flow = torch.rand(3*2,2,32,112,112)
# print('qq')
# out = model(X,flow)
# print(out)

# x = torch.tensor([[[1, 2, 3], [4, 5, 6]],[[1, 2, 3], [4, 5, 6]]])
# print(x.size())
# print(x)
# y = x.view(4,3)
# print(y.size())
# print(y)
# print(y.view(2,2,3))
