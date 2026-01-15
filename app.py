import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.title("📈 Portfolio Tracker (V1 + V2)")
st.caption("Loads holdings from portfolio.csv • Pulls market data via yfinance • Compares vs SPY")

# ---------- Load portfolio ----------
@st.cache_data
def load_portfolio(path="portfolio.csv"):
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    return df

portfolio = load_portfolio()
st.subheader("Holdings (from portfolio.csv)")
st.dataframe(portfolio, use_container_width=True)

tickers = portfolio["ticker"].tolist()

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
period = st.sidebar.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1)
benchmark = st.sidebar.text_input("Benchmark", "SPY").upper().strip()

# ---------- Robust price fetch ----------
@st.cache_data
def get_close_series(ticker: str, period: str, retries=3, pause=1.0):
    for _ in range(retries):
        try:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if data is not None and len(data) > 0 and "Close" in data.columns:
                s = data["Close"].dropna()
                if len(s) > 0:
                    return s
        except Exception:
            pass
        time.sleep(pause)
    return None

@st.cache_data
def get_prices(tickers, period):
    prices = pd.DataFrame()
    failed = []
    for t in tickers:
        s = get_close_series(t, period)
        if s is None:
            failed.append(t)
        else:
            prices[t] = s
    prices = prices.dropna(axis=1, how="all")
    return prices, failed

prices, failed = get_prices(tickers, period)

if failed:
    st.warning(f"These tickers returned no data and were excluded: {failed}")

st.write("Tickers used:", list(prices.columns))

shares = portfolio.set_index("ticker")["shares"]
shares = shares[shares.index.isin(prices.columns)].reindex(prices.columns)
prices = prices[shares.index]

spy = get_close_series(benchmark, period)
if spy is None:
    st.error(f"Failed to fetch benchmark {benchmark}. Try again.")
    st.stop()

# ---------- Portfolio value + returns ----------
position_values = prices.mul(shares, axis=1)
portfolio_value = position_values.sum(axis=1).dropna()

portfolio_returns = portfolio_value.pct_change().dropna()
spy_returns = spy.pct_change().dropna()

aligned = pd.concat([portfolio_returns, spy_returns], axis=1, join="inner").dropna()
aligned.columns = ["Portfolio", benchmark]

# ---------- V1 cumulative chart ----------
port_cum = (1 + aligned["Portfolio"]).cumprod()
bench_cum = (1 + aligned[benchmark]).cumprod()

# ---------- V2 metrics ----------
def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())

vol_port = float(aligned["Portfolio"].std() * np.sqrt(252))
vol_bench = float(aligned[benchmark].std() * np.sqrt(252))

dd_port = max_drawdown(aligned["Portfolio"])
dd_bench = max_drawdown(aligned[benchmark])

beta = float(np.cov(aligned["Portfolio"], aligned[benchmark])[0,1] / np.var(aligned[benchmark]))

summary = pd.DataFrame({
    "Metric": ["Annualized Volatility", "Max Drawdown", f"Beta vs {benchmark}"],
    "Portfolio": [vol_port, dd_port, beta],
    benchmark: [vol_bench, dd_bench, 1.0]
})

pretty = summary.copy()
mask = pretty["Metric"].isin(["Annualized Volatility", "Max Drawdown"])
pretty.loc[mask, "Portfolio"] = pretty.loc[mask, "Portfolio"].map(lambda x: f"{x*100:.1f}%")
pretty.loc[mask, benchmark] = pretty.loc[mask, benchmark].map(lambda x: f"{x*100:.1f}%")
pretty.loc[pretty["Metric"].str.contains("Beta"), "Portfolio"] = f"{beta:.2f}"
pretty.loc[pretty["Metric"].str.contains("Beta"), benchmark] = "1.00"

# ---------- Layout ----------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("V1: Cumulative Return")
    fig = plt.figure(figsize=(9,5))
    plt.plot(port_cum, label="Portfolio", linewidth=2)
    plt.plot(bench_cum, label=benchmark, linewidth=2)
    plt.title(f"Cumulative Return: Portfolio vs {benchmark}")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("V2: Risk Summary")
    st.dataframe(pretty, use_container_width=True)

st.divider()

st.subheader("V2: Correlation Matrix")
asset_returns = prices.pct_change().dropna()
corr = asset_returns.corr()

fig2, ax = plt.subplots(figsize=(9,7))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="Correlation")

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)

for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

ax.set_title("Asset Correlation Matrix (with values)")
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
# ---------- V3 Metrics (Return + Risk-adjusted) ----------
rf_annual = st.sidebar.number_input("Risk-free rate (annual, %)", value=4.0, step=0.5) / 100
rf_daily = (1 + rf_annual) ** (1/252) - 1

port = aligned["Portfolio"]
bench = aligned[benchmark]

# Annualized return (approx)
ann_return_port = (1 + port.mean()) ** 252 - 1
ann_return_bench = (1 + bench.mean()) ** 252 - 1

# Sharpe (excess return / vol)
ann_vol_port = port.std() * np.sqrt(252)
sharpe_port = ((port.mean() - rf_daily) / port.std()) * np.sqrt(252) if port.std() != 0 else np.nan

# Tracking error + Information Ratio
active = port - bench
tracking_error = active.std() * np.sqrt(252)
info_ratio = (active.mean() * 252) / tracking_error if tracking_error != 0 else np.nan

# Best / worst month (using monthly returns)
monthly = (1 + aligned).resample("M").prod() - 1
best_month = monthly["Portfolio"].max()
worst_month = monthly["Portfolio"].min()
