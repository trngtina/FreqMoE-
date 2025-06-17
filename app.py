import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data_processing import fetch_data, compute_log_returns, normalize_series, create_windows
from freqmoe_model import FreqMoE
from evaluation import directional_accuracy
from backtesting.backtest_strategy import generate_signals, backtest_returns
from backtesting.performance_metrics import compute_cumulative_returns, compute_sharpe_ratio, compute_max_drawdown, compute_hit_rate

st.set_page_config(page_title="FreqMoE-Finance Demo", layout="wide")

st.title("FreqMoE-Finance: Frequency-domain Mixture of Experts for Asset Forecasting")

st.markdown(
    """
    **Overview**: Upload a CSV of asset prices or select tickers to fetch data.
    Configure model hyperparameters, train/load a pretrained model, and visualize:
    - Price & log return series
    - Gating weights & frequency bands
    - Forecast vs. actual returns
    - Backtest performance (Sharpe, drawdown, hit rate)
    """
)

st.sidebar.header("1. Data Selection")
input_mode = st.sidebar.radio("Input Mode:", ["Upload CSV", "Fetch Tickers"])
if input_mode == "Upload CSV":
    csv_file = st.sidebar.file_uploader("Upload price CSV", type=["csv"])
    ticker_list = None
else:
    ticker_list = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,SPY")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "1m"])

st.sidebar.header("2. Model Hyperparameters")
num_experts = st.sidebar.slider("Number of Experts", 1, 8, 3, 1)
num_refine_blocks = st.sidebar.slider("Residual Blocks", 1, 4, 2, 1)
lookback = st.sidebar.slider("Lookback L", 32, 256, 64, 16)
horizon = st.sidebar.slider("Horizon H", 1, 64, 16, 1)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
lr = st.sidebar.select_slider("Learning Rate", options=[1e-5,1e-4,1e-3,1e-2], value=1e-3)
epochs = st.sidebar.slider("Epochs", 1, 50, 5, 1)

st.sidebar.header("3. Actions")
run_training = st.sidebar.button("Train & Forecast")
load_checkpoint = st.sidebar.file_uploader("Load Checkpoint (.pt)", type=["pt"])

if input_mode == "Upload CSV" and csv_file is not None:
    df_prices = pd.read_csv(csv_file, index_col=0, parse_dates=True).ffill().dropna()
    st.subheader("Uploaded Price Data Sample")
    st.dataframe(df_prices.head())
elif input_mode == "Fetch Tickers" and ticker_list:
    tickers = [t.strip().upper() for t in ticker_list.split(",")]
    df_prices = fetch_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), interval)
    st.subheader("Fetched Price Data Sample")
    st.dataframe(df_prices.head())
else:
    st.warning("Provide data via CSV or tickers.")
    st.stop()

df_returns = compute_log_returns(df_prices)
st.subheader("Log Returns (First 5 rows)")
st.dataframe(df_returns.head())

X_norm, means, stds = normalize_series(df_returns)
C, T = X_norm.shape

if T < lookback + horizon:
    st.error(f"Series too short (T={T}) for lookback={lookback}, horizon={horizon}.")
    st.stop()

x_input = X_norm[:, -lookback:]
x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FreqMoE(
    num_experts=num_experts,
    num_channels=C,
    lookback=lookback,
    horizon=horizon,
    num_refine_blocks=num_refine_blocks,
    dropout=0.3
).to(device)

if load_checkpoint is not None:
    ckpt = torch.load(load_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    st.success("Checkpoint loaded.")

if run_training:
    st.info("Training model...")
    x_windows, y_windows = create_windows(X_norm, lookback, horizon)
    N_train = min(500, x_windows.shape[0])
    x_train = x_windows[-N_train:]; y_train = y_windows[-N_train:]
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            y_pred_norm = model(xb)
            mu = torch.mean(xb, dim=-1, keepdim=True)
            sigma = torch.std(xb, dim=-1, keepdim=True)
            y_pred = y_pred_norm * sigma + mu

            y_true_norm = torch.zeros_like(y_pred_norm)
            for i in range(yb.shape[0]):
                for j in range(C):
                    y_true_norm[i, j] = (yb[i, j] - mu[i, j]) / (sigma[i, j] + 1e-6)
            loss = criterion(y_pred_norm, y_true_norm)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        st.write(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/N_train:.6f}")
    st.success("Training complete.")

model.eval()
with torch.no_grad():
    x_in = x_tensor.to(device)
    y_pred_norm = model(x_in)
    mu = torch.mean(x_in, dim=-1, keepdim=True)
    sigma = torch.std(x_in, dim=-1, keepdim=True)
    y_pred = y_pred_norm * sigma + mu
    y_pred = y_pred.squeeze(0).cpu().numpy()

last_date = df_returns.index[-1]
if 'd' in interval:
    freq_str = 'D'
elif 'h' in interval:
    freq_str = 'H'
else:
    freq_str = None
if freq_str:
    future_dates = pd.date_range(last_date, periods=horizon+1, freq=freq_str)[1:]
else:
    future_dates = pd.date_range(last_date, periods=horizon+1)[1:]

df_forecasts = pd.DataFrame(y_pred.T, index=future_dates, columns=df_returns.columns)
st.subheader("Forecasted Log Returns")
st.dataframe(df_forecasts)

st.subheader("Past Returns & Forecast")
fig, ax = plt.subplots(figsize=(10,4))
past_dates = df_returns.index[-lookback:]
for i, col in enumerate(df_returns.columns):
    ax.plot(past_dates, x_input[i], label=f"{col} (past)", linestyle="--", alpha=0.7)
    ax.plot(future_dates, df_forecasts[col], label=f"{col} (forecast)", linewidth=2)
ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("Log Return")
st.pyplot(fig)

st.subheader("Gating Weights & Frequency Bands")
Xf_last = torch.fft.rfft(x_in.squeeze(0), dim=-1)
masks = model.moe_block.freq_bands()
gate_logits = model.moe_block.gate(Xf_last.unsqueeze(0))
gate_weights = gate_logits.squeeze(0).cpu().numpy()

fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.bar(range(num_experts), gate_weights)
ax2.set_xticks(range(num_experts))
ax2.set_xlabel("Expert Index")
ax2.set_ylabel("Gate Weight")
ax2.set_title("Gate Weights for Last Window")
st.pyplot(fig2)

with st.expander("Show Frequency Band Boundaries"):
    raw_bounds = torch.sigmoid(model.moe_block.freq_bands().bound_params).cpu().detach().numpy()
    sorted_bounds = np.concatenate(([0.0], np.sort(raw_bounds), [1.0]))
    freq_boundaries = sorted_bounds * 0.5
    st.write("Normalized bounds (0 → Nyquist):", np.round(sorted_bounds, 3))
    st.write("Approx. frequency (cycles per timestep):", np.round(freq_boundaries, 4))

st.subheader("Backtest Performance")
df_ret_aligned = df_returns.reindex(df_forecasts.index, method="ffill")
signals = generate_signals(df_ret_aligned, df_forecasts)
strat_returns = backtest_returns(df_ret_aligned, signals)

cum_ret = compute_cumulative_returns(strat_returns)
sharpe = compute_sharpe_ratio(strat_returns, freq=252 if 'd' in interval else 252)
max_dd = compute_max_drawdown(strat_returns)
hit_rate = compute_hit_rate(strat_returns)

st.markdown(f"- **Annualized Sharpe Ratio**: {sharpe:.2f}")
st.markdown(f"- **Maximum Drawdown**: {max_dd:.2%}")
st.markdown(f"- **Hit Rate**: {hit_rate:.2%}")

fig3, ax3 = plt.subplots(figsize=(8,3))
ax3.plot(cum_ret.index, cum_ret.values, label="Strategy Cumulative Return")
ax3.set_xlabel("Date"); ax3.set_ylabel("Cumulative Return"); ax3.legend()
st.pyplot(fig3)
