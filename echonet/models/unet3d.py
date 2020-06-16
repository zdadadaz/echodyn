# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn as nn


# https://arxiv.org/pdf/1606.06650.pdf
# https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/model.py

class UNet3D(nn.Module):
    # acdc 112x112x112x3
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # multi-task
        # self.encoder_EF = UNet3D._block(features*2, features, name="decEF1")
        # self.conv_EF = nn.Conv3d(
        #     in_channels=features, out_channels=1, kernel_size=1
        # )
        # self.poolEF = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(56*56*1, 1)
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    # (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    # (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm3d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                    # (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    # (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


# +
class UNet3D_multi(nn.Module):
    # acdc 3x32x112x112
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D_multi, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # # multi-task
        # self.encoder_EF = UNet3D._block(features*2, features, name="decEF1")
        # self.conv_EF = nn.Conv3d(
        #     in_channels=features, out_channels=1, kernel_size=1
        # )
        # self.norm_EF = nn.InstanceNorm3d(num_features=features)
        # self.relu_EF = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.poolEF = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(56*56*16, 1)
        
#         self.enc_EF = {
#             2:nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), UNet3D._block(features, features * 2, name="enc2_ef")),
#             3:nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), UNet3D._block(features * 2, features * 4, name="enc3_ef")),
#             4:nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), UNet3D._block(features * 4, features * 8, name="enc4_ef")),
#             5:nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), UNet3D._block(features * 8, features * 16, name="bottleneck_ef"))
#         }
        
        self.fc = nn.Sequential(nn.Linear(480*2*7*7, 60*2*7*7),
                         nn.ReLU(),
                         nn.Linear(60*2*7*7, 1)
                        ) 
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # multi task
        
        Ef_out = self.fc(bottleneck.view(bottleneck.size(0),-1))
        
        return self.conv(dec1), Ef_out


# +
class UNet3D_ef(nn.Module):
    # acdc 3x32x112x112
    # echo 3x3x112x112
    def __init__(self, in_channels=3, out_channels=1, init_features=30):
        super(UNet3D_ef, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose3d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose3d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose3d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose3d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

#         self.conv = nn.Conv3d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )       
#         self.fc = nn.Linear(480*2*7*7, 1)
        self.fc = nn.Sequential(nn.Linear(480*2*7*7, 60*2*7*7),
                                 nn.ReLU(),
                                 nn.Linear(60*2*7*7, 1)
                                ) 
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
        
        Ef_out = self.fc(bottleneck.view(bottleneck.size(0),-1))
        
        return Ef_out


# -

class UNet3D_multi_1(nn.Module):
    # acdc 3x32x112x112
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D_multi_1, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.enc_EF = UNet3D._block(features * 8, features * 16, name="bottleneck_ef")
        
        self.fc = nn.Sequential(nn.Linear(480*2*7*7, 60*2*7*7),
                         nn.ReLU(),
                         nn.Linear(60*2*7*7, 1)
                        ) 
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # multi task
        bottleneck_ef = self.enc_EF(self.pool4(enc4))
        Ef_out = self.fc(bottleneck_ef.view(bottleneck_ef.size(0),-1))
        
        return self.conv(dec1), Ef_out


class UNet3D_multi_2(nn.Module):
    # acdc 3x32x112x112
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D_multi_2, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.enc_EF4 = UNet3D._block(features * 4, features * 8, name="enc4_ef")
        self.enc_EF5 = UNet3D._block(features * 8, features * 16, name="bottleneck_ef")
        self.fc = nn.Sequential(nn.Linear(480*2*7*7, 60*2*7*7),
                         nn.ReLU(),
                         nn.Linear(60*2*7*7, 1)
                        ) 
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # multi task

        bottleneck_ef = self.enc_EF4(self.pool3(enc3))
        bottleneck_ef = self.enc_EF5(self.pool4(bottleneck_ef))
        Ef_out = self.fc(bottleneck_ef.view(bottleneck_ef.size(0),-1))
        
        return self.conv(dec1), Ef_out


class UNet3D_multi_3(nn.Module):
    # acdc 3x32x112x112
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D_multi_3, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.enc_EF3 = UNet3D._block(features * 2, features * 4, name="enc3_ef")
        self.enc_EF4 = UNet3D._block(features * 4, features * 8, name="enc4_ef")
        self.enc_EF5 = UNet3D._block(features * 8, features * 16, name="bottleneck_ef")
        
        self.fc = nn.Sequential(nn.Linear(480*2*7*7, 60*2*7*7),
                         nn.ReLU(),
                         nn.Linear(60*2*7*7, 1)
                        ) 
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # multi task
        bottleneck_ef = self.enc_EF3(self.pool2(enc2))
        bottleneck_ef = self.enc_EF4(self.pool3(bottleneck_ef))
        bottleneck_ef = self.enc_EF5(self.pool4(bottleneck_ef))
        Ef_out = self.fc(bottleneck_ef.view(bottleneck_ef.size(0),-1))
        
        return self.conv(dec1), Ef_out


class UNet3D_multi_4(nn.Module):
    # acdc 3x32x112x112
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D_multi_4, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.enc_EF2 = UNet3D._block(features, features * 2, name="enc2_ef")
        self.enc_EF3 = UNet3D._block(features * 2, features * 4, name="enc3_ef")
        self.enc_EF4 = UNet3D._block(features * 4, features * 8, name="enc4_ef")
        self.enc_EF5 = UNet3D._block(features * 8, features * 16, name="bottleneck_ef")
        
        self.fc = nn.Sequential(nn.Linear(480*2*7*7, 60*2*7*7),
                         nn.ReLU(),
                         nn.Linear(60*2*7*7, 1)
                        ) 
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # multi task
        bottleneck_ef = self.enc_EF2(self.pool2(enc1))
        bottleneck_ef = self.enc_EF3(self.pool2(bottleneck_ef))
        bottleneck_ef = self.enc_EF4(self.pool3(bottleneck_ef))
        bottleneck_ef = self.enc_EF5(self.pool4(bottleneck_ef))
        Ef_out = self.fc(bottleneck_ef.view(bottleneck_ef.size(0),-1))
        
        return self.conv(dec1), Ef_out

class UNet3D_multi_opf(nn.Module):
    # acdc 3x32x112x112
    def __init__(self, in_channels=32, out_channels=1, init_features=30):
        super(UNet3D_multi_opf, self).__init__()
        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.enc_EF4 = UNet3D._block(features * 4, features * 8, name="enc4_ef")
        self.enc_EF5 = UNet3D._block(features * 8, features * 16, name="bottleneck_ef")
        self.fc = nn.Sequential(nn.Linear(480*2*7*7 * 2, 60*2*7*7),
                         nn.ReLU(),
                         nn.Linear(60*2*7*7, 1)
                        ) 
        
        # optical flow
        self.encoder1_opf = UNet3D._block(in_channels, features, name="enc1_opf")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2_opf = UNet3D._block(features, features * 2, name="enc2_opf")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3_opf = UNet3D._block(features * 2, features * 4, name="enc3_opf")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4_opf = UNet3D._block(features * 4, features * 8, name="enc4_opf")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck_opf = UNet3D._block(features * 8, features * 16, name="bottleneck_opf")

        self.upconv4_opf = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_opf = UNet3D._block((features * 8) * 2, features * 8, name="dec4_opf")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_opf = UNet3D._block((features * 4) * 2, features * 4, name="dec3_opf")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_opf = UNet3D._block((features * 2) * 2, features * 2, name="dec2_opf")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_opf = UNet3D._block(features * 2, features, name="dec1_opf")

        self.conv_opf = nn.Conv3d(
            in_channels=features, out_channels=2, kernel_size=1
        )
        
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        
        # optical flow
        enc1_opf = self.encoder1(x)
        enc2_opf = self.encoder2(self.pool1(enc1_opf))
        enc3_opf = self.encoder3(self.pool2(enc2_opf))
        enc4_opf = self.encoder4(self.pool3(enc3_opf))

        bottleneck_opf = self.bottleneck(self.pool4(enc4_opf))

        dec4_opf = self.upconv4(bottleneck_opf)
        dec4_opf = torch.cat((dec4_opf, enc4_opf), dim=1)
        dec4_opf = self.decoder4(dec4_opf)
        dec3_opf = self.upconv3(dec4_opf)
        dec3_opf = torch.cat((dec3_opf, enc3_opf), dim=1)
        dec3_opf = self.decoder3(dec3_opf)
        dec2_opf = self.upconv2(dec3_opf)
        dec2_opf = torch.cat((dec2_opf, enc2_opf), dim=1)
        dec2_opf = self.decoder2(dec2_opf)
        dec1_opf = self.upconv1(dec2_opf)
        dec1_opf = torch.cat((dec1_opf, enc1_opf), dim=1)
        dec1_opf = self.decoder1(dec1_opf)
       
         # multi task
        bottleneck_ef = self.enc_EF4(self.pool3(enc3))
        bottleneck_ef = self.enc_EF5(self.pool4(bottleneck_ef))
        bottleneck_ef = torch.cat((bottleneck_ef,bottleneck_opf), dim=1)
        Ef_out = self.fc(bottleneck_ef.view(bottleneck_ef.size(0),-1))
        
        return self.conv(dec1), Ef_out, self.conv_opf(dec1_opf)

