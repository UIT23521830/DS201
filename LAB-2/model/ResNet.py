import torch
from torch import nn
import torch.nn.functional as F

def collate_fn(items: list[dict]) -> dict[torch.Tensor]:
    images = torch.stack([item["image"] for item in items], dim=0)
    labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
    return {"image": images, "label": labels}

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Nhánh chính
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Nhánh shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 21, input_channels: int = 3):
        super().__init__()

        # Layer đầu tiên
        self.initial_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=2,
            padding=3,
            bias=False
        )
        self.initial_batch_norm = nn.BatchNorm2d(64)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # block ResNet 
        self.conv2_x = self._make_layer(in_channels=64, out_channels=64, num_blocks=2, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3_x = self._make_layer(in_channels=64, out_channels=128, num_blocks=2, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv4_x = self._make_layer(in_channels=128, out_channels=256, num_blocks=2, stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv5_x = self._make_layer(in_channels=256, out_channels=512, num_blocks=2, stride=1)

        # Global Average Pooling và Fully Connected
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    # layer gồm nhiều ResNetBlock
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResNetBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(in_channels=out_channels, out_channels=out_channels, stride=1))
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer đầu
        out = self.initial_conv(x)
        out = self.initial_batch_norm(out)
        out = F.relu(out)
        out = self.initial_pool(out)

        # ResNet + MaxPooling
        out = self.conv2_x(out)
        out = self.maxpool2(out)

        out = self.conv3_x(out)
        out = self.maxpool3(out)

        out = self.conv4_x(out)
        out = self.maxpool4(out)

        out = self.conv5_x(out)

        # Global Average Pooling và Fully Connected
        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)
        return out
