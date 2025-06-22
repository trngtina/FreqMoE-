"""
freqmoe_model.py

Core implementation of the Frequency-domain Mixture of Experts (FreqMoE) model
with residual refinement blocks for time series forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexReLU(nn.Module):
    """Complex-valued ReLU: applies ReLU separately to real and imaginary parts."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class FrequencyBands(nn.Module):
    """Learnable frequency band boundaries that produce binary masks."""
    def __init__(self, num_experts: int, freq_bins: int):
        super().__init__()
        self.num_experts = num_experts
        self.freq_bins = freq_bins
        # raw boundary parameters (one less than experts)
        self.bound_params = nn.Parameter(torch.zeros(num_experts - 1))

    def forward(self) -> torch.Tensor:
        raw = torch.sigmoid(self.bound_params)
        all_bounds = torch.cat([
            raw.new_tensor([0.0]),
            raw,
            raw.new_tensor([1.0])
        ])
        sorted_bounds, _ = torch.sort(all_bounds)
        indices = (sorted_bounds * (self.freq_bins - 1)).long()
        masks = []
        for i in range(self.num_experts):
            start = indices[i].item()
            end   = indices[i+1].item() if i < self.num_experts - 1 else self.freq_bins
            m = torch.zeros(self.freq_bins, device=raw.device)
            if end > start:
                m[start:end] = 1.0
            masks.append(m)
        return torch.stack(masks, dim=0)  # (num_experts, freq_bins)


class ComplexLinear(nn.Module):
    """Complex-valued linear layer mapping C^Fin -> C^Fout."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_real   = nn.Parameter(torch.zeros(out_features))
        self.bias_imag   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x.real
        xi = x.imag
        B, C, Fin = xr.shape
        xr_flat = xr.view(B * C, Fin)
        xi_flat = xi.view(B * C, Fin)
        out_real = (self.weight_real @ xr_flat.t() - self.weight_imag @ xi_flat.t()).t() + self.bias_real
        out_imag = (self.weight_real @ xi_flat.t() + self.weight_imag @ xr_flat.t()).t() + self.bias_imag
        Fout = out_real.size(-1)
        out_real = out_real.view(B, C, Fout)
        out_imag = out_imag.view(B, C, Fout)
        return torch.complex(out_real, out_imag)


class GateNetwork(nn.Module):
    """Gating network producing softmax weights for experts."""
    def __init__(self, num_experts: int, freq_bins: int):
        super().__init__()
        self.lin = nn.Linear(freq_bins, num_experts)

    def forward(self, Xf: torch.Tensor) -> torch.Tensor:
        # Xf: (B, C, F)
        mag = torch.mean(torch.abs(Xf), dim=1)  # (B, F)
        logits = self.lin(mag)                  # (B, num_experts)
        return F.softmax(logits, dim=-1)        # (B, num_experts)


class FreqMoEBlock(nn.Module):
    """Single Frequency-domain Mixture of Experts block with increased capacity."""
    def __init__(self, num_experts: int, num_channels: int, lookback: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_channels = num_channels
        self.lookback     = lookback
        self.freq_bins    = (lookback // 2) + 1

        # equal-width hidden dimension twice the freq_bins
        hidden_dim = self.freq_bins * 2
        self.freq_bands = FrequencyBands(num_experts, self.freq_bins)
        self.experts    = nn.ModuleList([
            nn.Sequential(
                ComplexLinear(self.freq_bins, hidden_dim),
                ComplexReLU(),
                ComplexLinear(hidden_dim, self.freq_bins)
            ) for _ in range(num_experts)
        ])
        self.gate = GateNetwork(num_experts, self.freq_bins)

    def forward(self, X_time: torch.Tensor) -> torch.Tensor:
        # X_time: (B, C, L)
        B, C, L = X_time.shape
        Xf = torch.fft.rfft(X_time, dim=-1)  # (B, C, freq_bins)

        masks = self.freq_bands()            # (num_experts, freq_bins)
        expert_outs = []
        for i, expert in enumerate(self.experts):
            m = masks[i].view(1, 1, self.freq_bins).type(Xf.dtype)
            expert_outs.append(expert(Xf * m))  # (B, C, freq_bins)

        gate_w = self.gate(Xf)               # (B, num_experts)
        Xf_comb = torch.zeros_like(Xf)
        for i, Yi in enumerate(expert_outs):
            Xf_comb += Yi * gate_w[:, i].view(B, 1, 1)

        return torch.fft.irfft(Xf_comb, n=L, dim=-1)  # (B, C, L)


class ResidualRefineBlock(nn.Module):
    """Residual refinement block for forecasting beyond reconstruction."""
    def __init__(self, num_channels: int, lookback: int, horizon: int, dropout: float = 0.3):
        super().__init__()
        self.lookback = lookback
        self.horizon  = horizon
        self.F_in     = (lookback // 2) + 1
        self.F_out    = ((lookback + horizon) // 2) + 1

        self.up1      = ComplexLinear(self.F_in, self.F_out)
        self.act      = ComplexReLU()
        self.dropout = nn.Dropout(dropout)
        self.up2      = ComplexLinear(self.F_out, self.F_out)

    def forward(self, residual_time: torch.Tensor) -> torch.Tensor:
        # residual_time: (B, C, L)
        B, C, L = residual_time.shape
        Rf = torch.fft.rfft(residual_time, dim=-1)  # (B, C, F_in)

        H1 = self.up1(Rf)
        H1 = self.act(H1)
        # apply dropout separately
        real = self.dropout(H1.real)
        imag = self.dropout(H1.imag)
        H1 = torch.complex(real, imag)

        Rf_up = self.up2(H1)  # (B, C, F_out)
        return torch.fft.irfft(Rf_up, n=self.lookback + self.horizon, dim=-1)


class FreqMoE(nn.Module):
    """Full FreqMoE model stacking MoE block + residual refinements."""
    def __init__(
        self,
        num_experts: int,
        num_channels: int,
        lookback: int,
        horizon: int,
        num_refine_blocks: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lookback         = lookback
        self.horizon          = horizon
        self.num_experts      = num_experts
        self.num_channels     = num_channels

        self.moe_block        = FreqMoEBlock(num_experts, num_channels, lookback)
        self.refine_blocks    = nn.ModuleList([
            ResidualRefineBlock(num_channels, lookback, horizon, dropout)
            for _ in range(num_refine_blocks)
        ])

    def forward(self, X_time: torch.Tensor) -> torch.Tensor:
        # X_time: (B, C, L)
        B, C, L = X_time.shape
        # initial reconstruction
        X_recon = self.moe_block(X_time)  # (B, C, L)

        # placeholder for forecast
        y_pred = torch.zeros(B, C, self.horizon, device=X_time.device, dtype=X_time.dtype)
        residual = X_time - X_recon        # (B, C, L)

        for block in self.refine_blocks:
            corr_full = block(residual)    # (B, C, L+H)
            corr_rec  = corr_full[..., :self.lookback]
            corr_fct  = corr_full[..., self.lookback:]
            residual  = residual - corr_rec
            y_pred    = y_pred + corr_fct

        return y_pred  # (B, C, horizon)
