import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyBands(nn.Module):
    
    def __init__(self, num_experts: int, freq_bins: int):
        super(FrequencyBands, self).__init__()
        self.N = num_experts
        self.F = freq_bins
        self.bound_params = nn.Parameter(torch.zeros(self.N - 1))

    def forward(self):
        raw = torch.sigmoid(self.bound_params)       # (N−1,)
        all_bounds = torch.cat((torch.tensor([0.0], device=raw.device),
                                raw,
                                torch.tensor([1.0], device=raw.device)))
        sorted_bounds, _ = torch.sort(all_bounds)
        indices = (sorted_bounds * (self.F - 1)).long()  # (N+1,)

        masks = []
        for i in range(self.N):
            start_idx = indices[i].item()
            end_idx = indices[i + 1].item() if i < self.N - 1 else self.F
            m = torch.zeros(self.F, device=raw.device)
            if end_idx > start_idx:
                m[start_idx:end_idx] = 1.0
            masks.append(m)
        return torch.stack(masks, dim=0)  # (N, F)

class ComplexLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(ComplexLinear, self).__init__()
        self.W_re = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.W_im = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.b_re = nn.Parameter(torch.zeros(out_features))
        self.b_im = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor):
        # x: (B, C, F_in), complex64
        x_re = x.real.view(-1, x.size(-1))  # (B*C, F_in)
        x_im = x.imag.view(-1, x.size(-1))  # (B*C, F_in)

        out_re = (self.W_re @ x_re.t() - self.W_im @ x_im.t()).t()
        out_im = (self.W_re @ x_im.t() + self.W_im @ x_re.t()).t()

        out_re = out_re + self.b_re.unsqueeze(0)
        out_im = out_im + self.b_im.unsqueeze(0)

        B, C, _ = x.shape
        F_out = out_re.size(-1)
        out_re = out_re.view(B, C, F_out)
        out_im = out_im.view(B, C, F_out)
        return torch.complex(out_re, out_im)

class GateNetwork(nn.Module):

    def __init__(self, num_experts: int, freq_bins: int):
        super(GateNetwork, self).__init__()
        self.lin = nn.Linear(freq_bins, num_experts)

    def forward(self, Xf: torch.Tensor):
        mag = torch.mean(torch.abs(Xf), dim=1)  # (B, F)
        logits = self.lin(mag)                  # (B, N)
        return F.softmax(logits, dim=-1)        # (B, N)

class FreqMoEBlock(nn.Module):

    def __init__(self, num_experts: int, num_channels: int, lookback: int):
        super(FreqMoEBlock, self).__init__()
        self.N = num_experts
        self.C = num_channels
        self.L = lookback
        self.F = (self.L // 2) + 1

        self.freq_bands = FrequencyBands(self.N, self.F)
        self.experts = nn.ModuleList([
            nn.Sequential(
                ComplexLinear(self.F, self.F),
                nn.Lambda(lambda z: torch.complex(F.relu(z.real), F.relu(z.imag))),
                ComplexLinear(self.F, self.F)
            ) for _ in range(self.N)
        ])
        self.gate = GateNetwork(self.N, self.F)

    def forward(self, X_time: torch.Tensor):
        B, C, L = X_time.shape
        Xf = torch.fft.rfft(X_time, dim=-1)          # (B, C, F)
        masks = self.freq_bands()                    # (N, F)

        expert_outs = []
        for i, expert in enumerate(self.experts):
            m = masks[i].view(1, 1, self.F).to(Xf.device).type(Xf.dtype)
            Xi = Xf * m
            Yi = expert(Xi)
            expert_outs.append(Yi)

        gate_w = self.gate(Xf)                       # (B, N)
        Xf_comb = torch.zeros_like(Xf)
        for i, Yi in enumerate(expert_outs):
            gi = gate_w[:, i].view(B, 1, 1)
            Xf_comb += Yi * gi

        X_recon = torch.fft.irfft(Xf_comb, n=self.L, dim=-1)  # (B, C, L)
        return X_recon

class ResidualRefineBlock(nn.Module):

    def __init__(self, num_channels: int, lookback: int, horizon: int, dropout: float = 0.3):
        super(ResidualRefineBlock, self).__init__()
        self.C = num_channels
        self.L = lookback
        self.H = horizon
        self.F_in = (self.L // 2) + 1
        self.F_out = ((self.L + self.H) // 2) + 1

        self.up1 = ComplexLinear(self.F_in, self.F_out)
        self.act = nn.Lambda(lambda z: torch.complex(F.relu(z.real), F.relu(z.imag)))
        self.dropout = nn.Dropout(dropout)
        self.up2 = ComplexLinear(self.F_out, self.F_out)

    def forward(self, residual_time: torch.Tensor):
        B, C, L = residual_time.shape
        Rf = torch.fft.rfft(residual_time, dim=-1)     # (B, C, F_in)
        H1 = self.up1(Rf)                              # (B, C, F_out)
        H1 = self.act(H1)
        H1 = self.dropout(H1)
        Rf_up = self.up2(H1)                           # (B, C, F_out)
        correction_full = torch.fft.irfft(Rf_up, n=self.L + self.H, dim=-1)
        return correction_full                         # (B, C, L+H)

class FreqMoE(nn.Module):

    def __init__(self, num_experts: int, num_channels: int, lookback: int, horizon: int, num_refine_blocks: int = 3, dropout: float = 0.3):
        super(FreqMoE, self).__init__()
        self.L = lookback
        self.H = horizon
        self.C = num_channels
        self.N = num_experts
        self.K = num_refine_blocks

        self.moe_block = FreqMoEBlock(self.N, self.C, self.L)
        self.refine_blocks = nn.ModuleList([
            ResidualRefineBlock(self.C, self.L, self.H, dropout=dropout)
            for _ in range(self.K)
        ])

    def forward(self, X_time: torch.Tensor):
        B, C, L = X_time.shape
        assert C == self.C and L == self.L

        X_recon = self.moe_block(X_time)              # (B, C, L)
        y_pred = torch.zeros(B, C, self.H, device=X_time.device, dtype=X_time.dtype)
        residual = X_time - X_recon                   # (B, C, L)

        for block in self.refine_blocks:
            corr_full = block(residual)               # (B, C, L+H)
            corr_rec = corr_full[:, :, :self.L]       # (B, C, L)
            corr_fct = corr_full[:, :, self.L:]       # (B, C, H)

            residual = residual - corr_rec
            y_pred = y_pred + corr_fct

        return y_pred
