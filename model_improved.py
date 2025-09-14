import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse your existing semi-analytical ODE block
# (keep the same import path as your project)
from odegcn import ODEG

############################################################
# Building Blocks
############################################################

class SqueezeExcite1D(nn.Module):
    """SE-style channel gating on (B, N, T, C) by pooling over N,T."""
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        hidden = max(4, channels // r)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, C)
        b, n, t, c = x.shape
        pooled = x.mean(dim=(1, 2))  # (B, C)
        gate = self.mlp(pooled).view(b, 1, 1, c)
        return x * gate


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-branch temporal conv on (B, N, T, C_in) -> (B, N, T, C_out)
    Each branch uses different kernel sizes and dilations, then fused.
    """
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.1,
                 kernels=(2, 3, 5), dilations=(1, 2, 4)):
        super().__init__()
        branches = []
        for k in kernels:
            for d in dilations:
                pad = (k - 1) * d
                conv = nn.Conv2d(c_in, c_out, kernel_size=(1, k), dilation=(1, d), padding=(0, pad))
                conv.weight.data.normal_(0, 0.01)
                branches.append(nn.Sequential(conv, Chomp1d(pad), nn.ReLU(inplace=True)))
        self.branches = nn.ModuleList(branches)
        self.proj = nn.Sequential(
            nn.Conv2d(len(branches) * c_out, c_out, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.down = None
        if c_in != c_out:
            self.down = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
            self.down.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, T, C) -> (B, C, N, T)
        y = x.permute(0, 3, 1, 2)
        feats = [br(y) for br in self.branches]
        ycat = torch.cat(feats, dim=1)
        out = self.proj(ycat)
        if self.down is not None:
            out = out + self.down(y)
        out = F.relu(out)
        return out.permute(0, 2, 3, 1)


class AdaptiveGCN(nn.Module):
    """
    Graph layer with learnable adaptive adjacency fused with a base A_hat.
    A_comb = sigma(g0)*A_hat + sigma(g1)*Softmax(ReLU(E1 @ E2^T))
    Then Y = ReLU(A_comb @ X @ Theta).
    X shape: (B, N, T, C_in)
    """
    def __init__(self, num_nodes: int, c_in: int, c_out: int, A_hat: torch.Tensor, emb_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer('A_hat', A_hat)
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out))
        stdv = 1.0 / math.sqrt(c_out)
        self.theta.data.uniform_(-stdv, stdv)

        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.g0 = nn.Parameter(torch.tensor(0.0))
        self.g1 = nn.Parameter(torch.tensor(0.0))
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, T, C_in)
        B, N, T, C = X.shape
        assert N == self.num_nodes
        # adaptive adjacency
        logits = F.relu(self.E1 @ self.E2.t())      # (N, N)
        A_adp = F.softmax(logits, dim=-1)           # row-normalized
        A_comb = torch.sigmoid(self.g0) * self.A_hat + torch.sigmoid(self.g1) * A_adp
        # GCN propagation
        y = torch.einsum('ij,bjtc->bitc', A_comb, X)           # (B, N, T, C)
        y = torch.einsum('bitc,cd->bitd', y, self.theta)       # (B, N, T, C_out)
        return F.relu(self.drop(y))


class AttentionFusionGate(nn.Module):
    """Fuse two tensors (B, N, T, C) with channel-wise gates in [0,1]."""
    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # pool both
        p1 = x1.mean(dim=(1, 2))  # (B, C)
        p2 = x2.mean(dim=(1, 2))  # (B, C)
        g = self.mlp(torch.cat([p1, p2], dim=-1))  # (B, C)
        g = g.view(g.size(0), 1, 1, g.size(-1))
        return g * x1 + (1.0 - g) * x2


############################################################
# Spatio-Temporal Block (Improved)
############################################################

class STBlockImproved(nn.Module):
    def __init__(self, num_nodes: int, c_in: int, c_hidden: int, A_hat: torch.Tensor,
                 odeg_steps: int = 12, odeg_solver: str = 'rk4', dropout: float = 0.1):
        super().__init__()
        self.tcn1 = MultiScaleTemporalConv(c_in, c_hidden, dropout=dropout)
        # Branch 1: ODEG on base graph
        self.odeg = ODEG(c_hidden, 12, A_hat, time=1.0, steps=odeg_steps, solver=odeg_solver)
        # Branch 2: Adaptive GCN on learned graph
        self.adp = AdaptiveGCN(num_nodes, c_hidden, c_hidden, A_hat, emb_dim=16, dropout=dropout)
        # Fuse
        self.fuse = AttentionFusionGate(c_hidden)
        # Second temporal conv + SE
        self.tcn2 = MultiScaleTemporalConv(c_hidden, c_hidden, dropout=dropout)
        self.se = SqueezeExcite1D(c_hidden)
        self.bn = nn.BatchNorm2d(num_nodes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, T, C_in)
        h = self.tcn1(X)                 # (B, N, T, C_h)
        y1 = self.odeg(h)                # (B, N, T, C_h)
        y2 = self.adp(h)                 # (B, N, T, C_h)
        y = self.fuse(y1, y2)            # (B, N, T, C_h)
        y = self.tcn2(F.relu(y))         # (B, N, T, C_h)
        y = self.se(y)                   # channel gating
        # BatchNorm over nodes dimension like original (expects (B, N, C, T))
        y_bn = self.bn(y.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return y_bn


############################################################
# Decoder Head
############################################################

class GRUDecoder(nn.Module):
    """
    Simple GRU decoder: (B, N, T, C) -> (B, N, T_out)
    We run GRU over the temporal dimension per-node and map last hidden to T_out.
    """
    def __init__(self, c_in: int, hidden: int, t_out: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=c_in, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, t_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, C)
        B, N, T, C = x.shape
        x_r = x.reshape(B * N, T, C)
        out, h = self.gru(x_r)              # h: (1, B*N, hidden)
        h = self.drop(h[-1])                 # (B*N, hidden)
        y = self.proj(h)                     # (B*N, T_out)
        return y.view(B, N, -1)              # (B, N, T_out)


############################################################
# Improved ODEGCN
############################################################

class ODEGCN_Improved(nn.Module):
    def __init__(self, num_nodes: int, num_features: int,
                 num_timesteps_input: int, num_timesteps_output: int,
                 A_sp_hat: torch.Tensor, A_se_hat: torch.Tensor,
                 hidden_channels: int = 64, num_stacks: int = 2,
                 odeg_steps: int = 12, odeg_solver: str = 'rk4',
                 dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.t_in = num_timesteps_input
        self.t_out = num_timesteps_output

        # Spatial and Semantic streams (stacked)
        self.sp_blocks = nn.ModuleList([
            STBlockImproved(num_nodes, num_features if i == 0 else hidden_channels,
                            hidden_channels, A_sp_hat, odeg_steps, odeg_solver, dropout)
            for i in range(num_stacks)
        ])
        self.se_blocks = nn.ModuleList([
            STBlockImproved(num_nodes, num_features if i == 0 else hidden_channels,
                            hidden_channels, A_se_hat, odeg_steps, odeg_solver, dropout)
            for i in range(num_stacks)
        ])

        # Final cross-stream fusion
        self.cross_fuse = AttentionFusionGate(hidden_channels)

        # Decoder
        self.decoder = GRUDecoder(c_in=hidden_channels, hidden=hidden_channels, t_out=self.t_out, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T_in, F)
        h_sp = x
        h_se = x
        for blk in self.sp_blocks:
            h_sp = blk(h_sp)  # (B, N, T, H)
        for blk in self.se_blocks:
            h_se = blk(h_se)  # (B, N, T, H)

        h = self.cross_fuse(h_sp, h_se)    # (B, N, T, H)
        y = self.decoder(h)                # (B, N, T_out)
        return y


############################################################
# Minimal sanity test
############################################################
if __name__ == "__main__":
    B, N, T_in, F_in, T_out = 2, 10, 12, 3, 12
    A = torch.eye(N)
    net = ODEGCN_Improved(
        num_nodes=N, num_features=F_in,
        num_timesteps_input=T_in, num_timesteps_output=T_out,
        A_sp_hat=A, A_se_hat=A, hidden_channels=64, num_stacks=2
    )
    x = torch.randn(B, N, T_in, F_in)
    y = net(x)
    print("Output shape:", y.shape)  # (B, N, T_out)
