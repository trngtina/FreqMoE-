import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from freqmoe_model import FreqMoE
from data_processing import fetch_data, compute_log_returns, normalize_series, create_windows, train_val_test_split
from evaluation import directional_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Train FreqMoE on financial data")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to fetch (e.g., AAPL MSFT)")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV file")
    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--num_refine_blocks", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.csv_path:
        df_prices = pd.read_csv(args.csv_path, index_col=0, parse_dates=True).ffill().dropna()
    else:
        df_prices = fetch_data(args.tickers, args.start_date, args.end_date, args.interval)

    df_returns = compute_log_returns(df_prices)
    X_norm, means, stds = normalize_series(df_returns)
    C, T = X_norm.shape

    x_windows, y_windows = create_windows(X_norm, args.lookback, args.horizon)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train_val_test_split(x_windows, y_windows)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = FreqMoE(
        num_experts=args.num_experts,
        num_channels=C,
        lookback=args.lookback,
        horizon=args.horizon,
        num_refine_blocks=args.num_refine_blocks,
        dropout=0.3
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        total_train_da = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device); y_batch = y_batch.to(device)

            y_pred_norm = model(x_batch)
            mu = torch.mean(x_batch, dim=-1, keepdim=True)
            sigma = torch.std(x_batch, dim=-1, keepdim=True)
            y_pred = y_pred_norm * sigma + mu

            y_true_norm = torch.zeros_like(y_pred_norm)
            for i in range(y_batch.shape[0]):
                for j in range(C):
                    y_true_norm[i, j] = (y_batch[i, j] - mu[i, j]) / (sigma[i, j] + 1e-6)

            loss = criterion(y_pred_norm, y_true_norm)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_loss += loss.item() * x_batch.size(0)

            da_batch = directional_accuracy(y_pred, y_batch)
            total_train_da += da_batch * x_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_da = total_train_da / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        total_val_da = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device); y_batch = y_batch.to(device)
                y_pred_norm = model(x_batch)
                mu = torch.mean(x_batch, dim=-1, keepdim=True)
                sigma = torch.std(x_batch, dim=-1, keepdim=True)
                y_pred = y_pred_norm * sigma + mu

                y_true_norm = torch.zeros_like(y_pred_norm)
                for i in range(y_batch.shape[0]):
                    for j in range(C):
                        y_true_norm[i, j] = (y_batch[i, j] - mu[i, j]) / (sigma[i, j] + 1e-6)

                loss = criterion(y_pred_norm, y_true_norm)
                total_val_loss += loss.item() * x_batch.size(0)
                da_batch = directional_accuracy(y_pred, y_batch)
                total_val_da += da_batch * x_batch.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_da = total_val_da / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | Train DA: {avg_train_da:.4f} | "
              f"Val Loss: {avg_val_loss:.6f} | Val DA: {avg_val_da:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, f"freqmoe_epoch{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, ckpt_path)
            print(f"Saved improved checkpoint to {ckpt_path}")

    # Test evaluation
    model.eval()
    total_test_loss = 0.0
    total_test_da = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device); y_batch = y_batch.to(device)
            y_pred_norm = model(x_batch)
            mu = torch.mean(x_batch, dim=-1, keepdim=True)
            sigma = torch.std(x_batch, dim=-1, keepdim=True)
            y_pred = y_pred_norm * sigma + mu

            y_true_norm = torch.zeros_like(y_pred_norm)
            for i in range(y_batch.shape[0]):
                for j in range(C):
                    y_true_norm[i, j] = (y_batch[i, j] - mu[i, j]) / (sigma[i, j] + 1e-6)

            loss = criterion(y_pred_norm, y_true_norm)
            total_test_loss += loss.item() * x_batch.size(0)
            da_batch = directional_accuracy(y_pred, y_batch)
            total_test_da += da_batch * x_batch.size(0)

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    avg_test_da = total_test_da / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.6f} | Test DA: {avg_test_da:.4f}")

if __name__ == "__main__":
    main()
