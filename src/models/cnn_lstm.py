import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str, freeze: bool = True):
    name = name.lower()
    if name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        features = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 2048
    elif name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        features = backbone.features
        out_channels = 1280
    elif name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        features = backbone.features
        out_channels = 1280
    elif name == "vgg16":
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = backbone.features
        out_channels = 512
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    if freeze:
        for p in features.parameters():
            p.requires_grad = False
    return features, out_channels


class CnnLstm(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50", freeze_cnn: bool = True):
        super().__init__()
        self.cnn, out_channels = build_backbone(backbone, freeze=freeze_cnn)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(out_channels, 256, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, 3, x.size(3), x.size(4))
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)
