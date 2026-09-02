import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import math
import pdb
##########################################################################
## 抄的轻量化transformer的实现
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x    
# =============================================================================

# =============================================================================
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)   
    
# 主体部分
class Restormer_Encoder(nn.Module):
    def __init__(self,inp_channels=1,out_channels=1,dim=64,num_blocks=[4, 4],heads=[8, 8, 8],ffn_expansion_factor=2,bias=False,LayerNorm_type='WithBias'):

        super(Restormer_Encoder, self).__init__()
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3,stride=1, padding=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        return out_enc_level1

class Restormer_Decoder(nn.Module):
    def __init__(self,inp_channels=1,out_channels=1,dim=64,num_blocks=[4, 4],heads=[8, 8, 8],ffn_expansion_factor=2,bias=False,LayerNorm_type='WithBias'):
        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,stride=1, padding=1, bias=bias))
        self.sigmoid = nn.Sigmoid()              
    def forward(self, inp_img):
        inp_enc_level2 = self.reduce_channel(inp_img)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_dec = self.output(out_enc_level2)
        out_dec = self.sigmoid(out_dec)
        return out_dec


class TeMAL(nn.Module):
    """Text-guided Modality-aware Adversarial Learning (TeMAL).

    The paper uses separate adapters for the visual and text features and
    combines each adapted feature with its residual before fusing them.
    """
    def __init__(self, in_channels, out_channels):
        super(prompt, self).__init__()
        self.text_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, out_channels),
        )
        self.text_residual = nn.Linear(in_channels, out_channels)
        self.visual_pro = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.alpha_img = nn.Parameter(torch.zeros(1))
        self.alpha_text = nn.Parameter(torch.zeros(1))

    def forward(self, x, text_feature):
        if x.ndim != 4:
            raise ValueError(f"TeMAL visual input must be 4D, got shape {tuple(x.shape)}")
        if text_feature.ndim != 2:
            raise ValueError(
                f"TeMAL text input must be 2D (batch, embedding), "
                f"got shape {tuple(text_feature.shape)}"
            )
        if x.shape[0] != text_feature.shape[0]:
            raise ValueError(
                f"TeMAL batch mismatch: visual batch {x.shape[0]}, "
                f"text batch {text_feature.shape[0]}"
            )

        visual_feature = self.alpha_img * x + self.visual_pro(x)
        text_feature = (
            self.alpha_text * self.text_residual(text_feature)
            + self.text_mlp(text_feature)
        ).unsqueeze(-1).unsqueeze(-1)
        return visual_feature + text_feature

class PTFuseGenerator(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(fusion, self).__init__()
        self.encoder = Restormer_Encoder(inp_channels, out_channels, dim, num_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.decoder = Restormer_Decoder(inp_channels, out_channels, dim, num_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type)
        # The encoder produces dim channels per modality, so concatenation
        # gives 2 * dim channels at the TeMAL input.
        self.temal = TeMAL(512, 2 * dim)
    def forward(self, img_ir, img_vis,target_text):
        out_ir = self.encoder(img_ir)
        out_vis = self.encoder(img_vis)
        fused = torch.cat([out_ir, out_vis], dim=1)
        fused = self.temal(fused, target_text)
        out_dec = self.decoder(fused)
        # sim_score = F.cosine_similarity(out_ir, out_vis)
        incompatibility = 1 - F.cosine_similarity(
            out_ir.flatten(1), out_vis.flatten(1), dim=1
        )
        return out_dec, incompatibility


class ModalityDiscriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, 3, 2, 1))  
        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, 2, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 2, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, 1))
        self.bn3 = nn.BatchNorm2d(256)

        self.temal = TeMAL(512, 256)

        # 添加 Spectral Norm 到全连接层
        self.connected_layer1 = nn.utils.spectral_norm(nn.Linear(6*6*256, 512))
        self.connected_layer2 = nn.utils.spectral_norm(nn.Linear(512, 128))
        self.connected_layer3 = nn.utils.spectral_norm(nn.Linear(128, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, text):
        x = self.leaky_relu(self.sn_conv1(x))
        x = self.leaky_relu(self.bn1(self.sn_conv2(x)))
        x = self.leaky_relu(self.bn2(self.sn_conv3(x)))
        x = self.leaky_relu(self.bn3(self.sn_conv4(x)))
        x = self.temal(x, text)
        x = x.flatten(1)
        x = self.leaky_relu(self.connected_layer1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.connected_layer2(x))
        x = self.dropout(x)
        x = self.connected_layer3(x)
        return x


# Backward-compatible names used by the existing training entry point.
prompt = TeMAL
fusion = PTFuseGenerator
discriminator = ModalityDiscriminator