import torch
from torch import nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self,channels: int):
        super().__init__()

        #left flow
        self.left_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1,1),
            padding=0
        )

        #middle flow
        self.conv_1_1 = nn.Conv2d(
            in_channels=channels,   
            out_channels=channels,
            kernel_size=(1,1),
            padding=0
        )

        self.conv_1_2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3,3),
            padding=1
        )

        self.conv_2_1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1,1),
            padding=0
        )

        self.conv_2_2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(5,5),
            padding=2
        )

        #right flow
        self.right_pool = nn.MaxPool2d(
            kernel_size=(3,3),
            padding=1,
            ceil_mode=True
        )

        self.right_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1,1),
            padding=0
        )

    def forward(self, x: torch.Tensor):
        #left flow
        left_features = self.left_conv(x)
        
        #middle flow
        middle_1_features = self.conv_1_2(
            self.conv_1_1(x)
        )
        
        middle_2_features = self.conv_2_2(
            self.conv_2_1(x)
        )
        
        #right flow
        right_features = self.right_pool(
            self.right_conv(x)  
        )

        output = torch.cat([
            left_features,
            middle_1_features,
            middle_2_features,
            right_features
        ], dim=1)  

        return output

class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7,7),
            stride=2,
            padding=3
        )

        self.pooling_1 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=1,
            ceil_mode=True
        )

        self.conv_2_1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1,1),
            padding=0
        )

        self.conv_2_2 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3,3),
            padding=1
        )

        self.pooling_2 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=1,
            ceil_mode=True
        )

        self.inception_3a = InceptionBlock(channels=192)
        self.inception_3b = InceptionBlock(channels=256)

        self.pooling_3 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=1,
            ceil_mode=True
        )

        # tự vt
        self.inception_4a = InceptionBlock(channels=480)
        self.inception_4b = InceptionBlock(channels=512)
        self.inception_4c = InceptionBlock(channels=512)
        self.inception_4d = InceptionBlock(channels=512)
        self.inception_4e = InceptionBlock(channels=528)

        self.pooling_4 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=1,
            ceil_mode=True
        )

        self.inception_5a = InceptionBlock(channels=832)
        self.inception_5b = InceptionBlock(channels=832)

        self.avg_pooling = nn.AvgPool2d(
            kernel_size=(7,7),
            stride=1
        )

        self.output = nn.Linear(
            in_features=832,
            out_features=10
        )

    def forward(self, images: torch.Tensor):
        # sử dụng softmax thay vì sigmoid
        features = F.relu(self.conv_1(images))
        features = self.pooling_1(features)

        features = F.relu(self.conv_2_1(features))
        features = F.relu(self.conv_2_2(features))
        features = self.pooling_2(features)

        features = self.inception_3a(features)
        features = self.inception_3b(features)
        features = self.pooling_3(features)

        features = self.inception_4a(features)
        features = self.inception_4b(features)
        features = self.inception_4c(features)
        features = self.inception_4d(features)
        features = self.inception_4e(features)
        features = self.pooling_4(features)

        features = self.inception_5a(features)
        features = self.inception_5b(features)
        features = self.avg_pooling(features)
        
        features = torch.flatten(features, start_dim=1)
        
        # output = F.softmax(output, dim=1)
        output = self.output(features)
        return output