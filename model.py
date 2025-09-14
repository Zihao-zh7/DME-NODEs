# model.py
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from odegcn import ODEG

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2**i
            in_c = num_inputs if i==0 else num_channels[i-1]
            out_c = num_channels[i]
            padding = (kernel_size-1)*dilation
            conv = nn.Conv2d(in_c, out_c, (1,kernel_size),
                             dilation=(1,dilation), padding=(0,padding))
            conv.weight.data.normal_(0,0.01)
            chomp = Chomp1d(padding)
            layers += [nn.Sequential(conv, chomp, nn.ReLU(), nn.Dropout(dropout))]
        self.network    = nn.Sequential(*layers)
        if num_inputs!=num_channels[-1]:
            self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1,1))
            self.downsample.weight.data.normal_(0,0.01)
        else:
            self.downsample = None

    def forward(self, x):
        # x: (B, N, T, F) -> (B, F, N, T)
        y = x.permute(0,3,1,2)
        out = self.network(y)
        if self.downsample:
            out = out + self.downsample(y)
        out = F.relu(out)
        return out.permute(0,2,3,1)

class GCN(nn.Module):
    def __init__(self, A_hat, in_c, out_c):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_c,out_c))
        stdv = 1./math.sqrt(out_c)
        self.theta.data.uniform_(-stdv,stdv)
    def forward(self, X):
        y = torch.einsum('ij,kjlm->kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm,mn->kjln', y, self.theta))

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_channels, num_nodes, A_hat):
        super(STGCNBlock, self).__init__()
        self.A_hat    = A_hat
        self.temporal1 = TemporalConvNet(in_c, out_channels)
        self.odeg     = ODEG(out_channels[-1], 12, A_hat, time=1.0, steps=12, solver='rk4')
        self.temporal2 = TemporalConvNet(out_channels[-1], out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        t = self.temporal1(X)           # (B, N, T, C1)
        t = self.odeg(t)                # (B, N, T, C1)
        t = self.temporal2(F.relu(t))   # (B, N, T, C2)
        return self.batch_norm(t)

class ODEGCN(nn.Module):
    def __init__(self, num_nodes, num_features,
                 num_timesteps_input, num_timesteps_output,
                 A_sp_hat, A_se_hat):
        super(ODEGCN, self).__init__()
        self.sp_blocks = nn.ModuleList([
            nn.Sequential(
                STGCNBlock(num_features,    [64,32,64], num_nodes, A_sp_hat),
                STGCNBlock(64,              [64,32,64], num_nodes, A_sp_hat)
            ) for _ in range(3)
        ])
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                STGCNBlock(num_features,    [64,32,64], num_nodes, A_se_hat),
                STGCNBlock(64,              [64,32,64], num_nodes, A_se_hat)
            ) for _ in range(3)
        ])
        self.pred = nn.Sequential(
            nn.Linear(num_timesteps_input*64, num_timesteps_output*32),
            nn.ReLU(),
            nn.Linear(num_timesteps_output*32, num_timesteps_output)
        )

    def forward(self, x):
        # x: (B, N, T_in, F)
        outs = []
        for blk in self.sp_blocks: outs.append(blk(x))
        for blk in self.se_blocks: outs.append(blk(x))
        x = torch.max(torch.stack(outs), dim=0)[0]      # (B, N, T, C)
        x = x.reshape(x.size(0), x.size(1), -1)         # (B, N, T*C)
        return self.pred(x)                             # (B, N, T_out)
