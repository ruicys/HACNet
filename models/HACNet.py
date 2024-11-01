import torch
import torch.nn as nn
import torch.nn.functional as F


class CAF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.initial_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        global_context = self.global_pool(x).view(x.size(0), -1)  # 全局上下文
        scale = self.fc(global_context).view(x.size(0), x.size(1), 1, 1)
        local_feature = self.local_conv(x)
        return local_feature * scale


#Context-Aware Adaptive Convolution (CAAC) Block
class CAACBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.GELU = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim * 3, dim * 4, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.caf = CAF(dim)
        self.small_receptive_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3,
                                              padding_mode='reflect')
        self.medium_receptive_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3,
                                               padding_mode='reflect')
        self.large_receptive_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3,
                                              padding_mode='reflect')

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.caf(x)
        small = self.small_receptive_conv(x)
        medium = self.medium_receptive_conv(x)
        large = self.large_receptive_conv(x)
        x = torch.cat([large, medium, small], dim=1)
        x = self.conv1(x)
        x = self.GELU(x)
        x = self.conv2(x)
        x = identity + x

        return x

class LG(nn.Module):
    def __init__(self, dim):
        super(LG, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim,
                      kernel_size=3,
                      padding=3 // 2,
                      groups=dim,
                      padding_mode='reflect')
        )
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_feature = self.local_conv(x)
        global_feature = self.global_conv(x)
        return local_feature * global_feature


class CALayer(nn.Module):
    def __init__(self, dim):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.ca(x)
        return x

class SALayer(nn.Module):
    def __init__(self, dim):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // 8, kernel_size=7, padding=3, bias=False)
        self.GELU = nn.GELU()
        self.conv2 = nn.Conv2d(dim // 8, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GELU(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x



class HAFBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.GELU = nn.GELU()
        self.conv1 = nn.Conv2d(dim * 3, dim * 4, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.lg = LG(dim)
        self.calayer = CALayer(dim)
        self.salayer = SALayer(dim)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        lg_x = self.lg(x)
        x = torch.cat([lg_x, self.calayer(x) * x, self.salayer(x) * x], dim=1)
        x = self.conv1(x)
        x = self.GELU(x)
        x = self.conv2(x)
        x = identity + x
        return x





#Hybrid Attention Context Network (HACNet)
class HACBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.caac = CAACBlock(dim)
        self.haf = HAFBlock(dim)

    def forward(self, x):
        x = self.caac(x)
        x = self.haf(x)
        return x



class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [HACBlock(dim=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class HACNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(HACNet, self).__init__()

        # setting
        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])


        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):

        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x)

        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def HACNet_t():
    return HACNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])

def HACNet_s():
    return HACNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[2, 2, 4, 2, 2])

def HACNet_b():
    return HACNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[4, 4, 8, 4, 4])

def HACNet_l():
    return HACNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[8, 8, 16, 8, 8])




