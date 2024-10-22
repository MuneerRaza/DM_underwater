import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##########################################################################
## Layer Norm
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
        return x / torch.sqrt(sigma + 1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

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

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
class EfficientSpatialAttention(nn.Module):
    def __init__(self, dim):
        super(EfficientSpatialAttention, self).__init__()

        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.sigmoid(self.conv(x))
        return x * attention_map
    
##########################################################################
class MultiScaleAttention(nn.Module):
    def __init__(self, num_heads, bias):
        super(MultiScaleAttention, self).__init__()
        self.num_heads = num_heads
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_sizes = [2*i + 3 for i in range(num_heads)]
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=bias) 
            for k_size in self.k_sizes
        ])
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Split input into multiple heads (channels are divided)
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        
        # Process each head with a different kernel size for multi-scale attention
        for i, head in enumerate(heads):
            y = self.avg_pool(head)
            y = self.convs[i](y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            out = head * y.expand_as(head)
            outputs.append(out)
        
        # Concatenate the outputs from all heads along the channel dimension
        output = torch.cat(outputs, dim=1)
        return output

## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.multiscale_attn = MultiScaleAttention(num_heads, bias)
        self.esa = EfficientSpatialAttention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.multiscale_attn(self.norm1(x)) * self.esa(x)
        x = x + self.ffn(self.norm2(x))
        return x

    
    
if __name__ == '__main__':
    input = torch.zeros([2, 48, 128, 128])
    # model = Restormer()
    # output = model(input)
    model2 = nn.Sequential(*[
        TransformerBlock(dim=int(48), num_heads=2, ffn_expansion_factor=2.66,
                         bias=False, LayerNorm_type='WithBias') for i in range(1)])
    # model3 = Attention_sa(1, 16, 48)
    output2 = model2(input)
    print(output2.shape)