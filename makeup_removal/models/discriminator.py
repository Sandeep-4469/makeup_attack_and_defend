import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):  # input (real_A) + target (real_B or fake_B)
        super(PatchDiscriminator, self).__init__()

        def conv_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, 64, norm=False),   # 128 → 64
            *conv_block(64, 128),                        # 64 → 32
            *conv_block(128, 256),                       # 32 → 16
            *conv_block(256, 512, stride=1),             # 16 → 16 (smaller receptive field)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 16 → 15 (approx 70x70 receptive field)
        )

    def forward(self, input_image, target_image):
        # Concatenate input and target images (real_A + real_B or real_A + fake_B)
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)
