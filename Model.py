import torch
import torch.nn as nn

'''
Resnet 9 is single input model,
Resnet 9 attention is dual input model
'''




class resnet9(nn.Module):
    def __init__(self, in_channels, feature_size, output_size):
        super().__init__()
        self.conv1 = self._block(in_channels, feature_size, pool=False)
        self.conv2 = self._block(feature_size, feature_size * 2, pool=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.res_block = nn.Sequential(
            self._block(feature_size * 2, feature_size * 2, pool=False),
            self._block(feature_size * 2, feature_size * 2, pool=False)
        )
        self.shortcut = nn.Conv2d(feature_size * 2, feature_size * 2, kernel_size=1, stride=1, padding=0)

        self.conv3 = self._block(feature_size * 2, feature_size * 4, pool=True)
        self.conv4 = self._block(feature_size * 4, feature_size * 8, pool=True)

        self.res_block2 = nn.Sequential(
            self._block(feature_size * 8, feature_size * 8, pool=False),
            self._block(feature_size * 8, feature_size * 8, pool=False)
        )
        self.shortcut2 = nn.Conv2d(feature_size * 8, feature_size * 8, kernel_size=1, stride=1, padding=0)

        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(feature_size * 8, output_size))


    def _block(self, in_channels, feature_size, pool=None):
        block_layers = [nn.Conv2d(in_channels, feature_size, 3, 2 if pool else 1, 1),
                        nn.BatchNorm2d(feature_size),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2)]
        return nn.Sequential(*block_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.res_block(x) + self.shortcut(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res_block2(x) + self.shortcut2(x)
        x = self.regression(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 2-channel input (average + max pooling)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return torch.sigmoid(attn)


class resnet9_with_attention(nn.Module):
    def __init__(self, in_channels, feature_size, num_classes):
        super().__init__()

        self.attention = SpatialAttention()

        self.conv1 = self._block(in_channels, feature_size)
        self.conv2 = self._block(feature_size, feature_size * 2, pool=True)

        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.res_block = nn.Sequential(self._block(feature_size * 2, feature_size * 2),
                                       self._block(feature_size * 2, feature_size * 2))

        self.shortcut1 = nn.Conv2d(feature_size * 2, feature_size * 2, 1, 1, 0)

        self.conv3 = self._block(feature_size * 2, feature_size * 4, pool=True)
        self.conv4 = self._block(feature_size * 4, feature_size * 8, pool=True)

        self.res_block_2 = nn.Sequential(self._block(feature_size * 8, feature_size * 8),
                                         self._block(feature_size * 8, feature_size * 8))

        self.shortcut2 = nn.Conv2d(feature_size * 8, feature_size * 8, 1, 1, 0)

        self.regression = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(feature_size * 8, num_classes),
                                        )

    def _block(self, in_channels, feature_size, pool=False):
        block_layers = [nn.Conv2d(in_channels, feature_size, 3, 2 if pool else 1, 1),
                        nn.BatchNorm2d(feature_size),
                        nn.ReLU(inplace=True)]
        return nn.Sequential(*block_layers)

    def forward(self, img):

        img1 = img[:, :3, :, :]  # First 3 channels
        img2 = img[:, 3:, :, :]  # Last 3 channels
        #
        attn_map = self.attention(img2)
        img2_attended = img2 * attn_map  #
        attn_map_1 = self.attention(img1)
        img1_attended = img1 * attn_map_1
        combined_input = torch.cat((img1_attended, img2_attended), dim=1)
        # combined_input = img1_attended

        x_1 = self.conv1(combined_input)
        x_2 = self.conv2(x_1)
        x_p = self.maxpool(x_2)

        x_all = self.res_block(x_p) + self.shortcut1(x_p)

        rest = self.conv3(x_all)
        rest = self.conv4(rest)
        rest = self.res_block_2(rest) + self.shortcut2(rest)

        rest = self.regression(rest)
        return rest




if __name__ == "__main__":
    pass
