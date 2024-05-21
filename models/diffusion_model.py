import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device)
                        * -(math.log(10000) / half_dim))
        emb = torch.outer(x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        assert h % 2 == 0 or w % 2 == 0, "w and h must be even"
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb):
        return self.upsample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_dim=None, activatedfun=nn.SiLU):
        super().__init__()
        self.time_dim = time_dim
        self.layer1 = nn.Sequential(
            activatedfun(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.layer2 = nn.Sequential(
            activatedfun(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        if time_dim:
            self.time_layer = nn.Sequential(
                activatedfun(),
                nn.Linear(time_dim, out_channels),
            )

        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        output = self.layer1(x)
        if self.time_dim:
            output += self.time_layer(time_emb)[:, :, None, None]
        output = self.layer2(output) + self.shortcut(x)

        return output
    
class Guided_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, guidance_channels, dropout, time_dim, activatedfun=nn.SiLU):
        super().__init__()
        self.layer1 = nn.Sequential(
            activatedfun(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.layer2 = nn.Sequential(
            activatedfun(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.time_layer = nn.Sequential(
            activatedfun(),
            nn.Linear(time_dim, out_channels),
        )

        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.conv1 = nn.Conv2d(guidance_channels, in_channels, 1)

        self.conv2 = nn.Conv2d(guidance_channels, out_channels, 1)

    def forward(self, x, gidance, time_emb):
        gidance1 = self.conv1(gidance)
        gidance2 = self.conv2(gidance)
        input = gidance1 + x
        output = self.layer1(input)
        output += self.time_layer(time_emb)[:, :, None, None] + gidance2
        output = self.layer2(output) + self.shortcut(x)

        return output
    
class Guidance(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, channel_mults=(1, 2, 3, 4), num_res_blocks=1, activatedfun=nn.SiLU, dropout=0.1):
        super().init()
        self.feature_model = nn.ModuleList()
        self.feature_model.append(nn.Conv2d(img_channels, base_channels, 3, padding=1)) 

        for _, mult in enumerate(channel_mults):
            self.out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.feature_model.append(
                    ResidualBlock(now_channels, self.out_channels, dropout, activatedfun=activatedfun))
                now_channels = self.out_channels

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = nn.Linear(self.out_channels, 1)
    
    def forward(self, x):
        feature_output = self.feature_model(x)
        feature_max = self.maxpool(feature_output).reshape(-1, self.out_channels)
        scale = self.mlp(self.maxpool(feature_output))
        return feature_max, scale
    
class UNet(nn.Module):
    def __init__(self, img_channels=9, base_channels=64, channel_mults=(1, 2, 3), num_res_blocks=2, time_dim=64 * 4, activatedfun=nn.SiLU, dropout=0.1):
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downblocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResidualBlock(now_channels, out_channels, dropout, time_dim=time_dim,
                                  activatedfun=activatedfun)
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downblocks.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim,
                          activatedfun=activatedfun),
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim,
                          activatedfun=activatedfun),
        ])

        self.upblocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResidualBlock(channels.pop() + now_channels, out_channels, dropout,
                                  time_dim=time_dim, activatedfun=activatedfun)
                )
                now_channels = out_channels

            if i != 0:
                self.upblocks.append(Upsample(now_channels))

        assert len(channels) == 0

        self.last_layer = nn.Sequential(
            activatedfun(),
            nn.Conv2d(base_channels, 3, 3, padding=1)
        )

    def forward(self, x, time):
        time_emb = self.time_embedding(time)
        x = self.init_conv(x)

        skips = [x]

        for layer in self.downblocks:
            x = layer(x, time_emb)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb)

        for layer in self.upblocks:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb)

        x = self.last_layer(x)
        assert len(skips) == 0
        return x

if __name__ == '__main__':
    # Debug
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = UNet(
        base_channels=64,
        channel_mults=(1, 2, 3),
        time_dim=256,
        num_res_blocks=2,
        dropout=0.0
        ).cuda()
    input = torch.randn(1, 6, 128, 128).cuda()
    time = torch.tensor([0.]).cuda()
    print("ratio",time)
    flops, params = profile(net, (input, time))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')