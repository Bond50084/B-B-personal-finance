from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

def get_stock_returns(ticker, start=None, end=None):
    """
    Fetch daily stock returns for a given company ticker.
    Default: last 8 years until today.
    """
    if end is None:
        end = pd.Timestamp.today().normalize()
    if start is None:
        start = end - pd.DateOffset(years=20)

    data = yf.download(ticker, start=start, end=end)

    # Flatten MultiIndex columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns.values]

    # Get adjusted close / close column
    price_cols = [c for c in data.columns if "Adj Close" in c or "Close" in c]
    if not price_cols:
        raise ValueError("No price column ('Close' or 'Adj Close') found in downloaded data.")

    col = price_cols[0]
    data["Daily Return"] = data[col].pct_change()

    return data


def plot_monthly_return_histogram(ticker, start=None, end=None):
    df = get_stock_returns(ticker, start, end)

    # Convert daily returns into monthly returns
    monthly_returns = (1 + df["Daily Return"]).resample("M").prod() - 1

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(monthly_returns.dropna(), bins=40, edgecolor="black", density=True, alpha=0.7)
    plt.title(f"Histogram of Monthly Returns for {ticker}\n(from {df.index.min().date()} to {df.index.max().date()})")
    plt.xlabel("Monthly Return")
    plt.ylabel("Probability Density")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# Example usage -> automatically uses last 8 years
plot_monthly_return_histogram("AAPL")
