from torch.nn import Module, Sequential, Linear, Embedding, SiLU, Conv2d, GroupNorm, Identity, ModuleList, Softmax, Upsample, Dropout, BatchNorm2d, Sigmoid, Tanh
from torch.optim import Adam
from torch import cat, tensor, randn, zeros, sin, cos, arange, exp, randint, square, sqrt as square_root, no_grad, min as minimum, max as maximum, ones, save, load
from torch.utils.data import DataLoader

from os.path import exists
import matplotlib.pyplot as plt

from math import log,sqrt

device = "cuda"

class ForwardProcess(Module):
    def __init__(self, betas):
        super().__init__()
        alphas = [1 - beta for beta in betas]
        alpha_bars = []
        for alpha in alphas:
            if len(alpha_bars) == 0:
                alpha_bars.append(alpha)
            else:
                alpha_bars.append(alpha_bars[-1] * alpha)
        alphas = tensor(alphas).requires_grad_(False)
        alpha_bars = tensor(alpha_bars).requires_grad_(False)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("betas", tensor(betas))

    def forward(self, X, t, device = "cpu"):
        noise = randn(*X.shape).to(device).requires_grad_(False)
        return square_root(self.alpha_bars[t]).view(X.shape[0], 1, 1, 1) * X + square_root(1 - self.alpha_bars[t]).view(X.shape[0], 1, 1, 1) * noise, noise


class TimeEncoding(Module):
    def __init__(self, time_dim, d_model, d_time):
        super().__init__()
        self.time_dim = time_dim
        self.d_model = d_model
        positional_encodings = zeros(time_dim, d_model)
        positions = arange(0, time_dim).float().unsqueeze(1)
        div_term = exp(arange(0, d_model, 2).float() * (-log(10000) / d_model))
        positional_encodings[:, 0::2] = sin(positions * div_term)
        positional_encodings[:, 1::2] = cos(positions * div_term)
        positional_encodings = positional_encodings
        self.temb = Sequential(
            Embedding.from_pretrained(positional_encodings),
            Linear(d_model, d_time),
            SiLU(),
            Linear(d_time, d_time)
        )
        
    def forward(self, t):
        return self.temb(t)


# In[11]:


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, d_time, self_attn, cross_attn, cond_dim):
        super().__init__()
        self.first_block = Sequential(
            GroupNorm(32, in_channels),
            SiLU(),
            Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.time_embedding = Sequential(
            SiLU(),
            Conv2d(d_time, out_channels, 1, 1, 0)
        )
        self.second_block = Sequential(
            GroupNorm(32, out_channels),
            SiLU(),
            Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.res = Conv2d(in_channels, out_channels, 1, 1, 0)
        self.proj = Conv2d(cond_dim, out_channels, 1, 1, 0)
        self.self_attn = self_attn
        self.cross_attn = cross_attn

    def forward(self, X, cond, temb):
        temb = self.time_embedding(temb.view(*temb.shape, 1, 1))
        res = self.res(X)
        out = self.first_block(X) + temb
        out = res + self.second_block(out)
        out = out + self.self_attn(out)
        cond = cond.view(*cond.shape, 1, 1)
        cond = self.proj(cond)
        if self.cross_attn is not None:
            return out + self.cross_attn(out, cond)
        return out


# In[12]:


class DownSample(Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.f_ = Sequential(
            GroupNorm(32, in_channels),
            Conv2d(in_channels, out_channels, 3, 2, 1),
            act
        )

    def forward(self, X):
        return self.f_(X)


# In[13]:


class DownBlock(Module):
    def __init__(self, res_block, down_block):
        super().__init__()
        self.res_block = res_block
        self.down_block = down_block

    def forward(self, X, cond, temb):
        X = self.res_block(X, cond, temb)
        return self.down_block(X)


# In[14]:


class SelfAttention(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.query = Conv2d(in_channels, out_channels, 1, 1, 0, bias = False)
        self.value = Conv2d(in_channels, out_channels, 1, 1, 0, bias = False)
        self.key = Conv2d(in_channels, out_channels, 1, 1, 0, bias = False)
        self.proj = Conv2d(out_channels, out_channels, 1, 1, 0, bias = False)
        self.softmax = Softmax(dim = -1)

    def forward(self, X):
        B, C, H, W = X.shape
        q = self.query(X)
        q = q.view(B, H * W, C)
        k = self.key(X)
        k = k.view(B, C, H * W)
        v = self.value(X)
        v = v.view(B, H * W, C)
        wei = q @ k / sqrt(self.out_channels)
        wei = self.softmax(wei)
        out = wei @ v
        out = out.view(B, C, H, W)
        return self.proj(out)


class CrossAttention(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.query = Conv2d(in_channels, out_channels, 1, 1, 0, bias = False)
        self.value = Conv2d(in_channels, out_channels, 1, 1, 0, bias = False)
        self.key = Conv2d(in_channels, out_channels, 1, 1, 0, bias = False)
        self.proj = Conv2d(out_channels, out_channels, 1, 1, 0, bias = False)
        self.softmax = Softmax(dim = -1)

    def forward(self, X, Y):
        B, C, H, W = X.shape
        q = self.query(X)
        q = q.view(B, H * W, C)
        k = self.key(Y)
        k = k.view(B, C, 1)
        v = self.value(Y)
        v = v.view(B, 1, C)
        wei = q @ k / sqrt(self.out_channels)
        wei = self.softmax(wei)
        out = wei @ v
        out = out.view(B, C, H, W)
        return self.proj(out)
# In[15]:


class UpSample(Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.f_ = Sequential(
            Upsample(scale_factor = 2, mode = "nearest"),
            Conv2d(in_channels, out_channels, 1, 1, 0),
            GroupNorm(32, out_channels),
            act
        )
        
    def forward(self, X):
        return self.f_(X)


# In[16]:


class UpBlock(Module):
    def __init__(self, up_block, res_block):
        super().__init__()
        self.up_block = up_block
        self.res_block = res_block

    def forward(self, X, Y, cond, temb):
        X = self.up_block(X)
        return self.res_block(cat([X, Y], dim = 1), cond, temb)


# In[17]:


class UNET(Module):
    def __init__(self, time_dim, d_model, d_time, init, in_channels, cond_dim):
        super().__init__()
        self.time_embedding = TimeEncoding(time_dim, d_model, d_time)
        self.conditional_embedding = Embedding(cond_dim, cond_dim)
        self.head = Conv2d(in_channels, init, 3, 1, 1)
        self.down_blocks = []
        mult = 1
        for i in range(3):
            self.down_blocks.append(DownBlock(
                ResidualBlock(init * mult, init * mult, d_time, SelfAttention(init * mult, init * mult) if i >= 1 else Identity(), CrossAttention(init * mult, init * mult) if i == 2 else None, cond_dim),
                DownSample(init * mult, init * mult * 2, SiLU())
            ))
            mult *= 2
        self.down_blocks = ModuleList(self.down_blocks)
        self.bottleneck = ModuleList([
            ResidualBlock(init * mult, init * mult, d_time, SelfAttention(init * mult, init * mult), CrossAttention(init * mult, init * mult), cond_dim),
            ResidualBlock(init * mult, init * mult, d_time, SelfAttention(init * mult, init * mult), CrossAttention(init * mult, init * mult), cond_dim)
        ])
        self.up_blocks = []
        for i in range(3):
            self.up_blocks.append(UpBlock(
                UpSample(init * mult, init * mult // 2, SiLU()),
                ResidualBlock(init * mult, init * mult // 2, d_time, SelfAttention(init * mult // 2, init * mult // 2) if i <= 1 else Identity(), CrossAttention(init * mult // 2, init * mult // 2) if i == 0 else None, cond_dim)
            ))
            mult //= 2
        self.up_blocks = ModuleList(self.up_blocks)
        self.tail = Conv2d(init, in_channels, 3, 1, 1)

    def forward(self, X, c, t):
        t = self.time_embedding(t)
        c = self.conditional_embedding(c)
        X = self.head(X)
        Y = [None for _ in range(len(self.down_blocks))]
        for i in range(len(self.down_blocks)):
            Y[i] = X.clone()
            X = self.down_blocks[i](X, c, t)
        for block in self.bottleneck:
            X = block(X, c, t)
        for i in range(len(self.up_blocks)):
            X = self.up_blocks[i](X, Y[len(self.down_blocks) - 1 - i], c, t)
        return self.tail(X)


# In[18]:


class DDPM(Module):
    def __init__(self, time_dim, d_model, d_time, init, in_channels, cond_dim):
        super().__init__()
        m = (0.02 - 1e-4) / (time_dim - 1)
        c = 1e-4 - m
        betas = [m * x + c for x in range(1, time_dim + 1)]
        self.time_steps = time_dim
        self.forw = ForwardProcess(betas)
        self.back = UNET(time_dim, d_model, d_time, init, in_channels, cond_dim)

    def forward(self, X, c, t, device):
        noisy_img, noise = self.forw(X, t, device)
        eps = self.back(noisy_img, c, t)
        return eps, noise

    def sample(self, shape, c, device):
        X = randn(shape).to(device)
        for t in range(self.time_steps, 0, -1):
            print ("\r" + str(t), end = "")
            z = randn(shape) if t > 1 else zeros(shape)
            alpha_t = self.forw.alphas[t - 1].to(device)
            alpha_bar_t = self.forw.alpha_bars[t - 1].to(device)
            beta_t = self.forw.betas[t - 1].to(device)
            prev_alpha_bar_t = self.forw.alpha_bars[t - 2] if t > 1 else self.forw.alphas[0]
            beta_bar_t = ((1 - prev_alpha_bar_t)/(1 - alpha_bar_t)) * beta_t
            sigma_t = square_root(beta_bar_t)
            time_tensor = (ones(shape[0]) * (t - 1)).to(device).int()
            X = 1 / square_root(alpha_t) * (X - (1 - alpha_t) / square_root(1 - alpha_bar_t) * self.back(X, c, time_tensor)) + sigma_t * z.to(device)
        X -= minimum(X)
        X /= maximum(X)
        return X
    

