import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------
# Upsampling Block (U-Net)
# ---------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------
# Braille Dot Detection Network
# ---------------------------
class BddNet(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
        self.input_size = input_size

        # -------- ResNet-50 Encoder --------
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # ---- Grayscale input (1 channel) ----
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        resnet.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.pool = resnet.maxpool                                      # ↓
        self.enc2 = resnet.layer1                                       # 256
        self.enc3 = resnet.layer2                                       # 512
        self.enc4 = resnet.layer3                                       # 1024
        self.enc5 = resnet.layer4                                       # 2048

        # -------- Bottleneck --------
        self.center = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # -------- Decoder (FIXED) --------
        self.up4 = UpBlock(1024, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)

        # -------- Final Heatmap --------
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        e1 = self.enc1(x)             # [B, 64,  H/2,  W/2]
        e2 = self.enc2(self.pool(e1)) # [B, 256, H/4,  W/4]
        e3 = self.enc3(e2)            # [B, 512, H/8,  W/8]
        e4 = self.enc4(e3)            # [B, 1024,H/16, W/16]
        e5 = self.enc5(e4)            # [B, 2048,H/32, W/32]

        # -------- Center --------
        c = self.center(e5)           # [B, 1024,H/32, W/32]

        # -------- Decoder --------
        x = self.up4(c, e4)           # [B, 512, H/16, W/16]
        x = self.up3(x, e3)           # [B, 256, H/8,  W/8]
        x = self.up2(x, e2)           # [B, 128, H/4,  W/4]
        x = self.up1(x, e1)           # [B, 64,  H/2,  W/2]

        x = self.final(x)             # [B, 1,   H/2,  W/2]

        # -------- Resize to GT size --------
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False
        )

        return torch.sigmoid(x)
