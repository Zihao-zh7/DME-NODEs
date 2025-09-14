import numpy as np
import torch
from torch import nn

# 是否使用 adjoint 方法（保留接口）
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class ModulationNet(nn.Module):
    """生成动态调制系数的轻量网络"""
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super(ModulationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x_mean: torch.Tensor) -> torch.Tensor:
        return self.net(x_mean)  # (B, F)


class ODEFunc(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj   = adj
        self.w     = nn.Parameter(torch.eye(feature_dim))
        self.d     = nn.Parameter(torch.ones(feature_dim))
        self.w2    = nn.Parameter(torch.eye(temporal_dim))
        self.d2    = nn.Parameter(torch.ones(temporal_dim))
        self.alpha = nn.Parameter(torch.zeros(feature_dim))
        self.modnet = ModulationNet(feature_dim, hidden_dim=128)
        self.register_buffer('modulation_scale', torch.tensor(0.5))

    def compute_base(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ij,kjlm->kilm', self.adj, x) - x

    def compute_modulation(self, x: torch.Tensor) -> torch.Tensor:
        d  = torch.clamp(self.d,  0, 1)
        W_h = torch.mm(self.w * d, (self.w * d).t())
        nl1 = torch.einsum('ijkl,lm->ijkm', x, W_h)

        d2 = torch.clamp(self.d2, 0, 1)
        W_t = torch.mm(self.w2 * d2, (self.w2 * d2).t())
        _ = torch.einsum('ijkl,km->ijml', nl1, W_t)  # 用作动态扰动

        x_mean = torch.mean(x, dim=(1, 2))  # (B, F)
        dyn = self.modnet(x_mean)
        alpha = self.alpha.unsqueeze(0) + dyn
        modulation = torch.tanh(alpha) * self.modulation_scale
        return modulation.unsqueeze(1).unsqueeze(1)  # (B,1,1,F)

    def forward(self, t, x):
        linear = self.compute_base(x)
        d  = torch.clamp(self.d,  0, 1)
        W_h = torch.mm(self.w * d, (self.w * d).t())
        nl1 = torch.einsum('ijkl,lm->ijkm', x, W_h)
        d2 = torch.clamp(self.d2, 0, 1)
        W_t = torch.mm(self.w2 * d2, (self.w2 * d2).t())
        nonlinear = torch.einsum('ijkl,km->ijml', nl1, W_t)
        beta = torch.sigmoid(self.alpha).unsqueeze(0).unsqueeze(1).unsqueeze(1)
        return linear + beta * torch.tanh(nonlinear)


class ExpODEblock(nn.Module):
    """原始 Euler-like 半解析积分"""
    def __init__(self, odefunc, steps=12, total_time=1.0):
        super(ExpODEblock, self).__init__()
        self.odefunc = odefunc
        self.steps   = steps
        self.delta   = total_time / steps

    def forward(self, x):
        h = x
        for _ in range(self.steps):
            f_base     = self.odefunc.compute_base(h)
            modulation = self.odefunc.compute_modulation(h)
            h = h + self.delta * f_base * torch.exp(modulation)
        return h


class RK4ExpODEblock(nn.Module):
    """4阶 Runge-Kutta 半解析积分"""
    def __init__(self, odefunc, steps=12, total_time=1.0):
        super(RK4ExpODEblock, self).__init__()
        self.odefunc = odefunc
        self.steps   = steps
        self.delta   = total_time / steps

    def _f(self, h):
        f_base     = self.odefunc.compute_base(h)
        modulation = self.odefunc.compute_modulation(h)
        return f_base * torch.exp(modulation)

    def forward(self, x):
        h = x
        for _ in range(self.steps):
            k1 = self._f(h)
            k2 = self._f(h + 0.5 * self.delta * k1)
            k3 = self._f(h + 0.5 * self.delta * k2)
            k4 = self._f(h + self.delta * k3)
            h  = h + (self.delta / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return h


class ODEG(nn.Module):
    """ 半解析 + 指数式积分（支持 rk4 / euler） """
    def __init__(self, feature_dim, temporal_dim, adj, time=1.0, steps=12, solver='rk4'):
        super(ODEG, self).__init__()
        odefunc = ODEFunc(feature_dim, temporal_dim, adj)
        if solver == 'rk4':
            self.block = RK4ExpODEblock(odefunc, steps=steps, total_time=time)
        else:
            self.block = ExpODEblock(odefunc, steps=steps, total_time=time)

    def forward(self, x):
        return self.block(x)
