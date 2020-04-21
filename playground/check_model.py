
from echonet.models.unet_brain import UNet, UNet_multi
from echonet.models.unet3d import UNet3D,UNet3D_multi, UNet3D_multi_4
from echonet.models.deeplabv3 import DeepLabV3_main, DeepLabV3_multi_main
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision
# import simulation
torch.cuda.empty_cache() 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Generate some random images
# input_images, target_masks = simulation.generate_random_data(112, 112, count=3)
# input_images = torch.rand(10,3,112,112)
input_images = torch.rand(2,3,32,112,112)

# model = UNet_multi(in_channels=3, out_channels=1)
# model = UNet(in_channels=3, out_channels=1)
# model = DeepLabV3_main()
# model = DeepLabV3_multi_main()
model = UNet3D_multi_4(in_channels=3, out_channels=1)
# model = torchvision.models.__dict__["resnet50"](
#         pretrained=False,
#         replace_stride_with_dilation=[False, True, True])
# model = torch.nn.Sequential(*(list(model.children())[:-1]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = model.to(device)
input_images = input_images.to(device)
# with torch.set_grad_enabled(False):
#     out = model(input_images)
#     print(out.shape)

summary(model, (3,32,112,112))
# summary(model, (3,112,112))
# summary(model, (1,28,28))


# with tqdm.tqdm(total=len(dataloader)) as pbar:
#     for (i, (_, (large_frame, small_frame, large_trace, small_trace, ef, esv, edv))) in enumerate(dataloader):

