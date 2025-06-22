import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data_processing import fetch_data, compute_log_returns, normalize_series, create_windows
from freqmoe_model import FreqMoE
from backtesting.backtest_strategy import generate_signals, backtest_returns
from backtesting.performance_metrics import (
    compute_cumulative_returns,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_hit_rate,
)

# ————— Streamlit Page Config —————
st.set_page_config(page_title="FreqMoE-Finance", layout="wide")
st.title("FreqMoE-Finance: Frequency-domain Mixture of Experts for Asset Forecasting")

# ————— 1) Sidebar: Data Input —————
st.sidebar.header("1. Data Input")
input_mode = st.sidebar.radio("Input Mode:", ["Upload CSV", "Fetch Tickers"], index=1)

df_prices = None
if input_mode == "Upload CSV":
    csv_file = st.sidebar.file_uploader("Upload price CSV", type=["csv"])
    interval="1d"
    if csv_file:
        df_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True).ffill().dropna()
elif input_mode == "Fetch Tickers":
    tickers    = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,SPY")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date   = st.sidebar.date_input("End Date",   pd.to_datetime("2025-06-22"))
    interval   = st.sidebar.selectbox("Interval", ["1d", "1h", "1m"])
    if tickers:
        try:
            df_prices = fetch_data(
                [t.strip().upper() for t in tickers.split(",")],
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval,
                tz="US/Eastern",
            )
        except ValueError as e:
            st.error(f"Data download error:\n{e}\nPlease check your tickers or upload a CSV.")

if df_prices is None:
    st.info("Awaiting data… please select or upload above.")
    st.stop()

# Show price sample
tick_cols = list(df_prices.columns)
st.subheader("Uploaded Price Data Sample")
st.dataframe(df_prices.head(), use_container_width=True)

# ————— 2) Sidebar: Settings —————
st.sidebar.header("2. Settings")
num_experts   = st.sidebar.slider("Number of Experts",    1, 8, 3)
num_refine    = st.sidebar.slider("Residual Blocks",      1, 4, 2)
lookback      = st.sidebar.slider("Lookback Window (L)", 32, 256, 64, step=16)
horizon       = 1 #st.sidebar.slider("Forecast Horizon (H)", 1, 64, 1)
backtest_days = st.sidebar.number_input("Backtest Holdout Days", 1, 100, 30)
batch_size    = st.sidebar.selectbox("Batch Size", [16,32,64,128], index=1)
lr            = st.sidebar.selectbox("Learning Rate", [1e-5,1e-4,1e-3,1e-2], index=2)
epochs        = st.sidebar.slider("Epochs", 1, 50, 5)
#hidden_mult      = st.sidebar.slider("Hidden-dimension multiplier", 1, 10, 2)
lambda_entropy   = st.sidebar.slider("Gate Entropy Penalty λ", 0.0, 0.3, 0.01, step=0.005)

# ————— 3) Data Prep & Holdout —————
df_returns = compute_log_returns(df_prices)
X_norm, means, stds = normalize_series(df_returns)
C, T = X_norm.shape

if T < lookback + horizon + backtest_days:
    st.error(f"Not enough data: need at least L+H+backtest_days = {lookback+horizon+backtest_days}, have {T}.")
    st.stop()

x_windows, y_windows = create_windows(X_norm, lookback, horizon)
Nw      = x_windows.shape[0]
N_train = Nw - backtest_days

x_train = x_windows[:N_train]
y_train = y_windows[:N_train]
x_test  = x_windows[N_train:]

# DataLoader
train_ds     = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# ————— 4) Model Setup & Training —————
# Gate-entropy penalty weight

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = FreqMoE(num_experts, C, lookback, horizon, num_refine).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

prog   = st.sidebar.progress(0)
status = st.sidebar.empty()
for ep in range(epochs):
    model.train()
    tot_loss = 0.0
    tot_ent  = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        # forward
        y_pred_norm = model(xb)
        # normalize
        mu    = xb.mean(dim=-1, keepdim=True)
        sigma = xb.std(dim=-1,  keepdim=True)
        yb_norm = (yb - mu) / (sigma + 1e-6)
        # compute base loss
        base_loss = criterion(y_pred_norm, yb_norm)
        # gate-entropy penalty
        Xf_batch = torch.fft.rfft(xb, dim=-1)
        gate_w   = model.moe_block.gate(Xf_batch)
        entropy  = - (gate_w * torch.log(gate_w + 1e-9)).sum(dim=-1).mean()
        loss     = base_loss - lambda_entropy * entropy
        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # stats
        tot_loss += base_loss.item() * xb.size(0)
        tot_ent  += entropy.item() * xb.size(0)
    avg_loss = tot_loss / N_train
    avg_ent  = tot_ent  / N_train
    prog.progress((ep+1)/epochs)
    status.text(f"Epoch {ep+1}/{epochs} — Loss: {avg_loss:.6f} — Entropy: {avg_ent:.4f}")
prog.empty()
status.success("Training complete")
status.empty()

# Display gate weight distribution on recent window
st.subheader("Gate Weights on Recent Window")
# prepare recent input if not already defined
x_recent = torch.tensor(X_norm[:, -lookback:], dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    Xf_recent = torch.fft.rfft(x_recent, dim=-1)
    gw_recent = model.moe_block.gate(Xf_recent).cpu().detach().numpy().squeeze(0)
labels = [f"Expert {i+1}" for i in range(num_experts)]
df_gate_recent = pd.DataFrame({'GateWeight': gw_recent}, index=labels)
st.bar_chart(df_gate_recent)
model.eval()
x_recent = torch.tensor(X_norm[:, -lookback:], dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    y_norm_full = model(x_recent)
mu_r    = x_recent.mean(dim=-1, keepdim=True)
sigma_r = x_recent.std(dim=-1,  keepdim=True)
y_full  = (y_norm_full * sigma_r + mu_r).squeeze(0).cpu().detach().numpy()

_, Hf = y_full.shape
freqstr = 'D' if interval.endswith('d') else ('H' if interval.endswith('h') else None)
future_idx = pd.date_range(df_returns.index[-1], periods=Hf+1, freq=freqstr)[1:]
df_full_fore = pd.DataFrame(y_full.T, index=future_idx, columns=tick_cols)

st.subheader(f"Full Forecast (Next {Hf} Steps)")
st.dataframe(df_full_fore, use_container_width=True)

fig_f, ax_f = plt.subplots(figsize=(10,4))
past_idx = df_returns.index[-lookback:]
for i, col in enumerate(tick_cols):
    ax_f.plot(past_idx,    x_recent[0,i].cpu().numpy(), "--", alpha=0.7, label=f"{col} past")
    ax_f.plot(future_idx, df_full_fore[col],       "-",  label=f"{col} forecast")
ax_f.set_xlabel("Date"); ax_f.set_ylabel("Log Return"); ax_f.legend(loc="best")
st.pyplot(fig_f)

# ————— Expert Frequency Bands & Parameters —————
st.subheader("Expert Frequency Bands & Parameters")

freq_bins = model.moe_block.freq_bands.freq_bins
# Use equal-width bands for clearer separation
eq_bounds = np.linspace(0.0, 1.0, num_experts+1)
indices   = (eq_bounds * (freq_bins - 1)).astype(int)

info_rows = []
for i in range(num_experts):
    lower     = eq_bounds[i]
    upper     = eq_bounds[i+1]
    idx_lower = indices[i]
    idx_upper = indices[i+1]
    num_bins  = idx_upper - idx_lower
    expert_mod = model.moe_block.experts[i]
    param_count = sum(p.numel() for p in expert_mod.parameters())
    info_rows.append({
        "Expert":      f"Expert {i+1}",
        "LowerBound":  round(lower,3),
        "UpperBound":  round(upper,3),
        "LowerIdx":    int(idx_lower),
        "UpperIdx":    int(idx_upper),
        "NumBins":     int(num_bins),
        "ParamCount":  int(param_count)
    })

df_info = pd.DataFrame(info_rows).set_index("Expert")
#st.dataframe(df_info, use_container_width=True)

# Plot expert band boundaries
st.subheader("Expert Band Boundaries Chart")
fig_band, ax_band = plt.subplots(figsize=(6, num_experts*0.5))
y = np.arange(num_experts)
for i in range(num_experts):
    ax_band.hlines(y[i], eq_bounds[i], eq_bounds[i+1], lw=8)
    ax_band.text((eq_bounds[i]+eq_bounds[i+1])/2, y[i], f"E{i+1}", va='center', ha='center', color='white')
ax_band.set_yticks(y)
ax_band.set_yticklabels([f"Expert {i+1}" for i in range(num_experts)])
ax_band.set_xlabel("Normalized Frequency Band")
st.pyplot(fig_band)

# ————— 6) Backtest vs Actual & Delta —————
st.subheader(f"Backtest vs Actual & Delta (Last {backtest_days} Points)")

# 1) Define holdout dates
dt = df_returns.index[-backtest_days:]

# 2) Generate forecasts on held-out windows
with torch.no_grad():
    y_norm_bt = model(x_test.to(device))
mu_bt    = x_test.mean(dim=-1, keepdim=True)
sigma_bt = x_test.std(dim=-1,  keepdim=True)
y_bt     = (y_norm_bt * sigma_bt + mu_bt).cpu().detach().numpy()
# y_bt shape: (days, 1, channels) when horizon=1
if y_bt.ndim == 3 and y_bt.shape[1] == 1:
    y_bt = y_bt.squeeze(1)   # now (days, channels)

# 3) Build forecast DataFrame using the original tickers
df_bt_fore = pd.DataFrame(y_bt, index=dt, columns=tick_cols)

# 4) Actual returns aligned to forecasted tickers
df_bt_act = df_returns.loc[dt, tick_cols]

# 5) Delta
df_delta = df_bt_fore - df_bt_act

# 6) Combine forecast, actual, delta
df_comb = pd.concat([
    df_bt_fore.add_suffix("_forecast"),
    df_bt_act .add_suffix("_actual"),
    df_delta .add_suffix("_delta"),
], axis=1)
df_comb.index.name = "Date"

# Display combined table
st.dataframe(df_comb, use_container_width=True)

# ————— 7) Expert Gate Weights —————) Expert Gate Weights —————) Expert Gate Weights —————
st.subheader("Expert Gate Weights During Backtest")
with torch.no_grad():
    Xf_bt   = torch.fft.rfft(x_test.to(device), dim=-1)
    weights = model.moe_block.gate(Xf_bt).cpu().detach().numpy()
labels = [f"Expert {i+1}" for i in range(num_experts)]
df_gate = pd.DataFrame(weights, index=dt, columns=labels)
st.dataframe(df_gate, use_container_width=True)
st.line_chart(df_gate)

# ————— 8) Backtest Performance —————
signals   = generate_signals(df_bt_act, df_bt_fore)
strat_ret = backtest_returns(df_bt_act, signals)
cum_ret   = compute_cumulative_returns(strat_ret)
sharpe    = compute_sharpe_ratio(strat_ret, freq=252)
max_dd    = compute_max_drawdown(strat_ret)
hit_rate  = compute_hit_rate(strat_ret)

st.markdown("**Backtest Metrics**")
st.markdown(f"- Sharpe Ratio: {sharpe:.2f}")
st.markdown(f"- Max Drawdown: {max_dd:.2%}")
st.markdown(f"- Hit Rate: {hit_rate:.2%}")

fig_bt, ax_bt = plt.subplots(figsize=(8,3))
ax_bt.plot(cum_ret.index, cum_ret.values, label="Strategy")
ax_bt.set_xlabel("Date"); ax_bt.set_ylabel("Cumulative Return"); ax_bt.legend()
st.pyplot(fig_bt)

# ————— Expert Contributions to Backtest Forecast —————
st.subheader("Expert Contributions to Backtest Forecast")

# df_bt_fore: DataFrame of shape (backtest_days, tickers)
# df_gate:    DataFrame of shape (backtest_days, experts)

# Build a contributions DataFrame with a MultiIndex on columns:
#   level 0 = Expert name, level 1 = Ticker
contrib_dict = {}
for ex in df_gate.columns:
    # multiply each forecast series by that expert's gate weight
    contrib_dict[ex] = df_bt_fore.mul(df_gate[ex], axis=0)

# concat into one wide DataFrame
df_contrib = pd.concat(contrib_dict, axis=1)
# e.g. columns like ("Expert 1", "AAPL"), ("Expert 1", "MSFT"), ("Expert 2","AAPL"), …

# Show table
st.dataframe(df_contrib, use_container_width=True)

# And visualize it as a stacked area chart to see relative contributions
# (sum over tickers to get total contribution per expert if you like)
# Show the detailed contributions table
st.dataframe(df_contrib, use_container_width=True)

# Now aggregate per‐expert by summing across tickers
# (df_contrib has MultiIndex columns: (expert, ticker))
df_expert_sum = df_contrib.groupby(level=0, axis=1).sum()

# Display the aggregated time series
st.subheader("Expert Contribution Over Time")
st.area_chart(df_expert_sum)
