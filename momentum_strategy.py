"""
╔══════════════════════════════════════════════════════════════╗
║   QuantQuest – E-Summit '26, IIT Mandi                      ║
║   ML Momentum Strategy | Long-Only | Top-2 Weekly Rotation  ║
╚══════════════════════════════════════════════════════════════╝
Run in Jupyter or as a plain Python script.
pip install yfinance scikit-learn xgboost matplotlib seaborn pandas numpy
"""

# ── CELL 1 ─────────────────────────────────────────────────────
# IMPORTS
# ───────────────────────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found – using LR + RF ensemble only.")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F9F9F9",
    "axes.grid": True, "grid.alpha": 0.35,
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
})

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ── CELL 2 ─────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────
TICKERS      = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","JPM","V","JNJ","BRK-B"]
START        = "2017-01-01"
END          = "2025-12-31"
TRAIN_END    = "2022-12-31"
TEST_START   = "2023-01-01"
TOP_N        = 2           # stocks selected per week
TC           = 0.001       # 0.1% transaction cost (each side)
WEEK_FREQ    = "W-FRI"     # rebalance every Friday


# ── CELL 3 ─────────────────────────────────────────────────────
# DATA DOWNLOAD
# ───────────────────────────────────────────────────────────────
print("Downloading data …")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)

# Keep OHLCV; forward-fill gaps (holidays, missing days)
close  = raw["Close"].ffill()
high   = raw["High"].ffill()
low    = raw["Low"].ffill()
volume = raw["Volume"].ffill()
open_  = raw["Open"].ffill()

print(f"Data shape: {close.shape}  |  Tickers: {list(close.columns)}")
print(f"Date range: {close.index[0].date()} → {close.index[-1].date()}")
close.head(3)


# ── CELL 4 ─────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ───────────────────────────────────────────────────────────────
def make_features(close, high, low, volume):
    """Build a rich feature matrix for every (date, ticker) pair."""
    features = {}

    for tk in close.columns:
        c = close[tk]
        h = high[tk]
        lo = low[tk]
        v = volume[tk]

        df = pd.DataFrame(index=c.index)

        # ── Returns ──────────────────────────────────────────────
        df["ret_1d"]   = c.pct_change(1)
        df["ret_5d"]   = c.pct_change(5)
        df["ret_10d"]  = c.pct_change(10)
        df["ret_21d"]  = c.pct_change(21)
        df["ret_63d"]  = c.pct_change(63)
        df["ret_126d"] = c.pct_change(126)
        df["ret_252d"] = c.pct_change(252)

        # ── Momentum / skip-1-week ────────────────────────────────
        df["mom_4w"]   = c.pct_change(21) - c.pct_change(5)   # 4-week skipping last week
        df["mom_13w"]  = c.pct_change(63) - c.pct_change(5)

        # ── Volatility ───────────────────────────────────────────
        df["vol_5d"]   = df["ret_1d"].rolling(5).std()
        df["vol_21d"]  = df["ret_1d"].rolling(21).std()
        df["vol_63d"]  = df["ret_1d"].rolling(63).std()

        # ── Risk-adjusted momentum ────────────────────────────────
        df["sharpe_5d"]  = df["ret_5d"] / (df["vol_5d"] + 1e-9)
        df["sharpe_21d"] = df["ret_21d"] / (df["vol_21d"] + 1e-9)
        df["sharpe_63d"] = df["ret_63d"] / (df["vol_63d"] + 1e-9)

        # ── Moving averages & crossovers ─────────────────────────
        ma5  = c.rolling(5).mean()
        ma21 = c.rolling(21).mean()
        ma63 = c.rolling(63).mean()
        ma200= c.rolling(200).mean()

        df["price_vs_ma21"]  = c / ma21 - 1
        df["price_vs_ma63"]  = c / ma63 - 1
        df["price_vs_ma200"] = c / ma200 - 1
        df["ma5_vs_ma21"]    = ma5 / ma21 - 1
        df["ma21_vs_ma63"]   = ma21 / ma63 - 1

        # ── RSI (14-day) ──────────────────────────────────────────
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        df["rsi14"] = 100 - 100 / (1 + rs)

        # ── MACD ──────────────────────────────────────────────────
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd - sig
        df["macd_norm"] = df["macd_hist"] / (c + 1e-9)

        # ── Bollinger Band position ───────────────────────────────
        bb_mid  = c.rolling(20).mean()
        bb_std  = c.rolling(20).std()
        df["bb_pct"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

        # ── Volume features ───────────────────────────────────────
        df["vol_ratio_5_21"] = v.rolling(5).mean() / (v.rolling(21).mean() + 1e-9)
        df["vol_ratio_21_63"]= v.rolling(21).mean() / (v.rolling(63).mean() + 1e-9)
        df["obv"]            = (np.sign(c.diff()) * v).cumsum()
        df["obv_5d_chg"]     = df["obv"].pct_change(5)

        # ── ATR-based normalised range ────────────────────────────
        tr = pd.concat([
            h - lo,
            (h - c.shift()).abs(),
            (lo - c.shift()).abs()
        ], axis=1).max(axis=1)
        df["atr14_norm"] = tr.rolling(14).mean() / (c + 1e-9)

        # ── Skewness & kurtosis of returns ────────────────────────
        df["skew_21d"] = df["ret_1d"].rolling(21).skew()
        df["kurt_21d"] = df["ret_1d"].rolling(21).kurt()

        # ── Drawdown from rolling 63-day high ─────────────────────
        df["dd_from_63d_high"] = c / c.rolling(63).max() - 1

        features[tk] = df

    return features


feat_dict = make_features(close, high, low, volume)

# Build a stacked (date × ticker) DataFrame
all_feat = pd.concat(feat_dict, axis=1)   # MultiIndex columns
all_feat.columns.names = ["Ticker", "Feature"]
print(f"Feature matrix: {all_feat.shape}")


# ── CELL 5 ─────────────────────────────────────────────────────
# LABELS  — did the stock return > 0 over the NEXT 5 trading days?
# ───────────────────────────────────────────────────────────────
labels = {}
for tk in close.columns:
    fwd_ret = close[tk].pct_change(5).shift(-5)   # forward 5-day return
    labels[tk] = (fwd_ret > 0).astype(int)

labels_df = pd.DataFrame(labels)
print("Label (positive next-week return) prevalence:")
print(labels_df.mean().round(3))


# ── CELL 6 ─────────────────────────────────────────────────────
# RESAMPLE TO WEEKLY (every Friday close) for portfolio construction
# ───────────────────────────────────────────────────────────────
# For modelling we use DAILY data; for backtesting we sample weekly
weekly_close = close.resample(WEEK_FREQ).last().dropna(how="all")
weekly_ret   = weekly_close.pct_change().shift(-1)   # next-week return
print(f"Weekly periods: {len(weekly_close)}")


# ── CELL 7 ─────────────────────────────────────────────────────
# TRAIN / TEST SPLIT  (daily)
# ───────────────────────────────────────────────────────────────
feat_cols = feat_dict[TICKERS[0]].columns.tolist()

def build_Xy(tickers, feat_dict, labels_df, start=None, end=None):
    """Stack all tickers into one flat X, y for a date range."""
    X_parts, y_parts = [], []
    for tk in tickers:
        df  = feat_dict[tk].copy()
        lbl = labels_df[tk].copy()
        if start:
            df  = df.loc[start:]
            lbl = lbl.loc[start:]
        if end:
            df  = df.loc[:end]
            lbl = lbl.loc[:end]
        df["_ticker"] = tk
        df["_label"]  = lbl
        X_parts.append(df)
    combined = pd.concat(X_parts).dropna()
    y = combined["_label"].astype(int)
    X = combined.drop(columns=["_ticker","_label"])
    return X, y

X_train, y_train = build_Xy(TICKERS, feat_dict, labels_df,
                             start=START, end=TRAIN_END)
X_test,  y_test  = build_Xy(TICKERS, feat_dict, labels_df,
                             start=TEST_START, end=END)

print(f"Train: {X_train.shape}  |  positives: {y_train.mean():.2%}")
print(f"Test : {X_test.shape}   |  positives: {y_test.mean():.2%}")


# ── CELL 8 ─────────────────────────────────────────────────────
# MODEL BUILDING  — Ensemble: LR + RF + (XGB if available)
# ───────────────────────────────────────────────────────────────
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    CalibratedClassifierCV(
                   LogisticRegression(C=0.1, max_iter=1000,
                                      class_weight="balanced",
                                      random_state=RANDOM_STATE),
                   cv=3))
])

rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=30,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)

estimators = [("lr", lr), ("rf", rf)]

if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )
    estimators.append(("xgb", xgb))

ensemble = VotingClassifier(estimators=estimators, voting="soft")

print("Training ensemble …")
ensemble.fit(X_train, y_train)
print("Done.")

# ── Evaluation ──────────────────────────────────────────────────
y_prob_test = ensemble.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

print("\n── Test Set Classification Report ──────────────────────────")
print(classification_report(y_test, y_pred_test, digits=3))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_test):.4f}")


# ── CELL 9 ─────────────────────────────────────────────────────
# FEATURE IMPORTANCE  (from RF component)
# ───────────────────────────────────────────────────────────────
rf_fitted = ensemble.named_estimators_["rf"]
imp = pd.Series(rf_fitted.feature_importances_, index=feat_cols)
top20 = imp.nlargest(20)

fig, ax = plt.subplots(figsize=(9, 6))
top20.sort_values().plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Top-20 Feature Importances (Random Forest)", fontweight="bold")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/feature_importance.png", dpi=150)
plt.show()
print(top20.to_string())


# ── CELL 10 ────────────────────────────────────────────────────
# WEEKLY PREDICTION ENGINE
# ── For each Friday, predict P(positive next week) for all tickers
# ───────────────────────────────────────────────────────────────
def predict_weekly_probs(ensemble, feat_dict, weekly_dates, feat_cols):
    """
    For each Friday in weekly_dates, look up the feature row for
    each ticker on that date (or the last available prior date)
    and return a DataFrame of predicted probabilities.
    """
    records = []
    for dt in weekly_dates:
        row = {"Date": dt}
        for tk in TICKERS:
            df = feat_dict[tk][feat_cols]
            # find the closest available date ≤ dt
            avail = df.index[df.index <= dt]
            if len(avail) == 0:
                row[tk] = np.nan
                continue
            feat_row = df.loc[avail[-1]].values.reshape(1, -1)
            if np.isnan(feat_row).any():
                row[tk] = np.nan
            else:
                row[tk] = ensemble.predict_proba(feat_row)[0, 1]
        records.append(row)

    prob_df = pd.DataFrame(records).set_index("Date")
    return prob_df


# Run on ALL weeks (train + test) for full backtest
all_weekly_dates = weekly_close.index
print(f"Generating predictions for {len(all_weekly_dates)} weeks …")
prob_df = predict_weekly_probs(ensemble, feat_dict, all_weekly_dates, feat_cols)
print("Predictions done.")
print(prob_df.tail(5))


# ── CELL 11 ────────────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION & BACKTEST
# ───────────────────────────────────────────────────────────────
def backtest(prob_df, weekly_close, top_n=TOP_N, tc=TC):
    """
    Each week:
      1. Rank tickers by predicted P(positive).
      2. Select top_n; equal-weight.
      3. Compute gross weekly return from the NEXT week's close.
      4. Deduct tc on entry AND exit (0.2% round-trip total).
    Returns portfolio_returns, weights_df, selected_stocks_df
    """
    dates = prob_df.index
    port_rets    = []
    weights_list = []
    selected_list= []

    prev_portfolio = []   # track previous week's holdings for TC

    for i, dt in enumerate(dates[:-1]):       # last week has no forward return
        next_dt = dates[i + 1]

        probs = prob_df.loc[dt].dropna()
        ranked = probs.nlargest(top_n)
        selected = ranked.index.tolist()

        # Equal weight
        w = 1.0 / top_n

        # Get next-week return for selected stocks
        gross_rets = []
        for tk in selected:
            r = weekly_close.loc[next_dt, tk] / weekly_close.loc[dt, tk] - 1
            gross_rets.append(r)

        port_gross = np.mean(gross_rets)

        # Transaction costs:
        # Entry cost on new positions, exit cost on positions being closed
        new_positions  = set(selected) - set(prev_portfolio)
        exit_positions = set(prev_portfolio) - set(selected)
        # For simplicity: full rebalance every week = always pay both sides
        # (conservative; actual turnover may be lower)
        tc_cost = tc * len(new_positions) / top_n + tc * len(exit_positions) / top_n
        port_net = port_gross - tc_cost

        port_rets.append({
            "Date": next_dt,
            "Gross_Return": port_gross,
            "TC_Cost":      tc_cost,
            "Net_Return":   port_net,
        })

        weights_list.append({"Date": dt, **{tk: (w if tk in selected else 0) for tk in TICKERS}})
        selected_list.append({"Date": dt, "Selected": selected,
                               **{f"Prob_{tk}": prob_df.loc[dt, tk] for tk in TICKERS}})

        prev_portfolio = selected

    port_df  = pd.DataFrame(port_rets).set_index("Date")
    wts_df   = pd.DataFrame(weights_list).set_index("Date")
    sel_df   = pd.DataFrame(selected_list).set_index("Date")
    return port_df, wts_df, sel_df


port_df, wts_df, sel_df = backtest(prob_df, weekly_close)

# Add buy-and-hold SPY benchmark for reference
spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
spy_weekly = spy["Close"].resample(WEEK_FREQ).last().ffill().pct_change().dropna()

# Align to portfolio dates
bench = spy_weekly.reindex(port_df.index).fillna(0)
port_df["Benchmark"] = bench.values


# ── CELL 12 ────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ───────────────────────────────────────────────────────────────
WEEKS_PER_YEAR = 52

def perf_metrics(weekly_rets, label="Strategy"):
    wr   = weekly_rets.dropna()
    cum  = (1 + wr).cumprod()
    total_years = len(wr) / WEEKS_PER_YEAR

    ann_ret  = cum.iloc[-1] ** (1 / total_years) - 1
    ann_vol  = wr.std() * np.sqrt(WEEKS_PER_YEAR)
    sharpe   = ann_ret / (ann_vol + 1e-9)
    hit_rate = (wr > 0).mean()
    avg_wret = wr.mean()

    rolling_max = cum.cummax()
    drawdown    = cum / rolling_max - 1
    max_dd      = drawdown.min()

    total_ret = cum.iloc[-1] - 1

    metrics = {
        "Label":            label,
        "Cumulative Return":f"{total_ret:.2%}",
        "Annualised Return":f"{ann_ret:.2%}",
        "Annualised Vol":   f"{ann_vol:.2%}",
        "Sharpe Ratio":     f"{sharpe:.3f}",
        "Max Drawdown":     f"{max_dd:.2%}",
        "Hit Rate":         f"{hit_rate:.2%}",
        "Avg Weekly Ret":   f"{avg_wret:.4%}",
    }
    return metrics, cum, drawdown

# Full period
m_gross, cum_gross, dd_gross = perf_metrics(port_df["Gross_Return"], "Strategy (Gross)")
m_net,   cum_net,   dd_net   = perf_metrics(port_df["Net_Return"],   "Strategy (Net TC)")
m_bench, cum_bench, dd_bench = perf_metrics(port_df["Benchmark"],    "SPY Benchmark")

# Test period only
test_mask = port_df.index >= TEST_START
m_gross_t, cum_gross_t, _ = perf_metrics(port_df.loc[test_mask,"Gross_Return"],"Test Gross")
m_net_t,   cum_net_t,   _ = perf_metrics(port_df.loc[test_mask,"Net_Return"],  "Test Net")
m_bench_t, cum_bench_t, _ = perf_metrics(port_df.loc[test_mask,"Benchmark"],   "Test SPY")

metrics_table = pd.DataFrame([m_gross, m_net, m_bench])
metrics_table_test = pd.DataFrame([m_gross_t, m_net_t, m_bench_t])

print("\n── Full Period (2017–2025) ──────────────────────────────────")
print(metrics_table.set_index("Label").to_string())
print("\n── Test Period (2023–2025) ─────────────────────────────────")
print(metrics_table_test.set_index("Label").to_string())


# ── CELL 13 ────────────────────────────────────────────────────
# VISUALISATIONS
# ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
gs  = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

# ── 1. Cumulative Returns ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_gross.index, cum_gross.values, label="Strategy (Gross)", lw=2, color="#2196F3")
ax1.plot(cum_net.index,   cum_net.values,   label="Strategy (Net TC)",lw=2, color="#1565C0", ls="--")
ax1.plot(cum_bench.index, cum_bench.values, label="SPY Benchmark",   lw=2, color="#E53935", alpha=0.8)
ax1.axvline(pd.Timestamp(TEST_START), color="gray", ls=":", lw=1.5, label="Train/Test split")
ax1.fill_between(cum_gross.index,
                 [1]*len(cum_gross), cum_gross.values,
                 where=cum_gross.values >= 1, alpha=0.08, color="#2196F3")
ax1.set_title("Cumulative Portfolio Value ($1 Invested)", fontweight="bold")
ax1.set_ylabel("Portfolio Value")
ax1.legend(loc="upper left", fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ── 2. Drawdown ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(dd_net.index, dd_net.values, 0, alpha=0.4, color="#E53935", label="Net Drawdown")
ax2.fill_between(dd_bench.index, dd_bench.values, 0, alpha=0.25, color="#9E9E9E", label="SPY Drawdown")
ax2.axvline(pd.Timestamp(TEST_START), color="gray", ls=":", lw=1.5)
ax2.set_title("Drawdown", fontweight="bold")
ax2.set_ylabel("Drawdown")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ── 3. Weekly Returns Distribution ────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(port_df["Net_Return"].dropna(), bins=60, color="#2196F3", alpha=0.75, edgecolor="white", label="Net")
ax3.hist(port_df["Benchmark"].dropna(), bins=60, color="#E53935", alpha=0.45, edgecolor="white", label="SPY")
ax3.axvline(0, color="black", lw=1.2)
ax3.set_title("Weekly Return Distribution", fontweight="bold")
ax3.set_xlabel("Weekly Return")
ax3.legend(fontsize=9)

# ── 4. Rolling 26-week Sharpe ─────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
roll_sharpe = (
    port_df["Net_Return"].rolling(26).mean() /
    (port_df["Net_Return"].rolling(26).std() + 1e-9)
) * np.sqrt(WEEKS_PER_YEAR)
ax4.plot(roll_sharpe.index, roll_sharpe.values, color="#1565C0", lw=1.5)
ax4.axhline(0, color="black", lw=1, ls="--")
ax4.axhline(1, color="green", lw=1, ls="--", alpha=0.6, label="Sharpe=1")
ax4.axvline(pd.Timestamp(TEST_START), color="gray", ls=":", lw=1.5)
ax4.set_title("Rolling 26-Week Sharpe Ratio (Net)", fontweight="bold")
ax4.legend(fontsize=9)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ── 5. Stock Selection Frequency (test period) ────────────────
ax5 = fig.add_subplot(gs[3, 0])
sel_flat = sel_df.loc[test_mask, "Selected"].explode()
freq = sel_flat.value_counts()
bars = ax5.bar(freq.index, freq.values, color="#4CAF50", edgecolor="white", alpha=0.85)
ax5.bar_label(bars, padding=2, fontsize=8)
ax5.set_title("Stock Selection Frequency (Test 2023–2025)", fontweight="bold")
ax5.set_ylabel("Times Selected")
ax5.tick_params(axis="x", rotation=30)

# ── 6. Monthly heatmap of net returns ─────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
monthly = port_df["Net_Return"].resample("ME").apply(lambda x: (1+x).prod()-1)
monthly_df = monthly.to_frame("Return")
monthly_df["Year"]  = monthly_df.index.year
monthly_df["Month"] = monthly_df.index.month
pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
sns.heatmap(pivot, ax=ax6, cmap="RdYlGn", center=0, annot=True, fmt=".1%",
            linewidths=0.5, annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
ax6.set_title("Monthly Net Return Heatmap", fontweight="bold")
ax6.set_xlabel("")

plt.suptitle("QuantQuest ML Momentum Strategy – Full Backtest Report",
             fontsize=15, fontweight="bold", y=0.995)
plt.savefig("/mnt/user-data/outputs/backtest_report.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: backtest_report.png")


# ── CELL 14 ────────────────────────────────────────────────────
# PREDICTIONS CSV OUTPUT
# ───────────────────────────────────────────────────────────────
prob_cols = [f"Prob_{tk}" for tk in TICKERS]

output_csv = sel_df.copy()
# Rename prob columns to Prob_TICKER
for tk in TICKERS:
    if f"Prob_{tk}" not in output_csv.columns:
        output_csv[f"Prob_{tk}"] = prob_df[tk]

# Add selected stocks and weights
def get_weights(row):
    sel = row["Selected"]
    w   = 1 / TOP_N
    return {tk: (w if tk in sel else 0.0) for tk in TICKERS}

weight_rows = output_csv.apply(get_weights, axis=1, result_type="expand")
weight_rows.columns = [f"Weight_{tk}" for tk in TICKERS]

final_csv = pd.concat([
    output_csv[["Selected"] + [f"Prob_{tk}" for tk in TICKERS]],
    weight_rows,
    port_df[["Gross_Return", "Net_Return", "TC_Cost"]].reindex(output_csv.index)
], axis=1)

final_csv.index.name = "Date"
csv_path = "/mnt/user-data/outputs/weekly_predictions.csv"
final_csv.to_csv(csv_path)
print(f"Saved: {csv_path}")
print(final_csv.tail(10).to_string())


# ── CELL 15 ────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION  (optional advanced section)
# ──  Retrain every year, test on next year
# ───────────────────────────────────────────────────────────────
print("\n── Walk-Forward Validation ─────────────────────────────────")

wf_years  = list(range(2018, 2026))   # each is the TEST year
wf_results= []

for test_yr in wf_years:
    tr_start = "2017-01-01"
    tr_end   = f"{test_yr - 1}-12-31"
    te_start = f"{test_yr}-01-01"
    te_end   = f"{test_yr}-12-31"

    Xtr, ytr = build_Xy(TICKERS, feat_dict, labels_df, tr_start, tr_end)
    Xte, yte = build_Xy(TICKERS, feat_dict, labels_df, te_start, te_end)

    if len(Xte) == 0:
        continue

    # Lightweight model for walk-forward (RF only for speed)
    m = RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=30,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    # Portfolio for that year
    wk_dates_yr = weekly_close.loc[te_start:te_end].index
    p_yr = predict_weekly_probs(m, feat_dict, wk_dates_yr, feat_cols)
    port_yr, _, _ = backtest(p_yr, weekly_close.loc[te_start:te_end])
    metrics_yr, _, _ = perf_metrics(port_yr["Net_Return"], str(test_yr))

    wf_results.append({
        "Year": test_yr, "AUC": round(auc, 4),
        **{k: v for k, v in metrics_yr.items() if k != "Label"}
    })

wf_df = pd.DataFrame(wf_results).set_index("Year")
print(wf_df.to_string())
wf_df.to_csv("/mnt/user-data/outputs/walk_forward_results.csv")
print("Saved: walk_forward_results.csv")


# ── CELL 16 ────────────────────────────────────────────────────
# FINAL SUMMARY PRINT
# ───────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║              STRATEGY SUMMARY                               ║
╠══════════════════════════════════════════════════════════════╣
║  Universe   : 10 large-cap US equities                      ║
║  Signal     : ML Ensemble (LR + RF + XGB) predicted P(+)   ║
║  Selection  : Top-2 by predicted probability each week      ║
║  Portfolio  : Equal-weight, long-only                       ║
║  Rebalance  : Weekly (Friday close)                         ║
║  TC         : 0.1% entry + 0.1% exit (per position)        ║
║  Train      : 2017-01-01 → 2022-12-31                       ║
║  Test       : 2023-01-01 → 2025-12-31                       ║
╚══════════════════════════════════════════════════════════════╝
""")
print("FULL PERIOD METRICS")
print(metrics_table.set_index("Label").to_string())
print("\nTEST PERIOD METRICS")
print(metrics_table_test.set_index("Label").to_string())
print("\nAll outputs saved to /mnt/user-data/outputs/")
