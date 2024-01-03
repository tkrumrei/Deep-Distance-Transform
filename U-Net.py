"""
Wie sieht das Modell aus:
- in allen Convolutional Layern wird padding benutzt
- filter Kernels in the first convolutional Layern
- Best case: d = 4 channels = 16
- 
- Für 10 Epochen trainiert
- Adam optimizer
- biases initialized to 0
- ReLU 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceTransformUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, c=64, depth=3, activation='relu'):
        super(DistanceTransformUNet, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_channels = in_channels if i == 0 else c * 2**(i-1)
            out_channels = c * 2**i
            self.encoder.append(self.conv_block(in_channels, out_channels, activation))

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(depth, 0, -1):
            in_channels = c * 2**i
            out_channels = c * 2**(i-1)
            self.decoder.append(self.conv_block(in_channels, out_channels, activation))

        # Final convolution layer
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, activation):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU() if activation == 'relu' else nn.SELU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU() if activation == 'relu' else nn.SELU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        encoder_features = []
        for block in self.encoder:
            x = block(x)
            encoder_features.append(x)

        # Decoder
        for i, block in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, encoder_features[-(i + 1)]], dim=1)
            x = block(x)

        # Final convolution
        x = self.final_conv(x)

        return x

# Example usage
model = DistanceTransformUNet(in_channels=1, out_channels=1, c=64, depth=3, activation='relu')
print(model)