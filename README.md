\# ML Momentum Trading Strategy

\### QuantQuest Algorithmic Trading Challenge | E-Summit '26 | IIT Mandi



\---



\## Strategy Overview

A machine learning-based long-only momentum strategy that predicts weekly stock returns and constructs an optimized portfolio from a universe of 10 large-cap US equities.



\## Universe

`AAPL` `MSFT` `GOOGL` `AMZN` `META` `TSLA` `JPM` `V` `JNJ` `BRK-B`



\## Key Results

| Metric | Full Period | Test Period (2023–25) |

|---|---|---|

| Sharpe Ratio | 4.21 | 1.47 |

| Annualised Return | 118% | 31% |

| Max Drawdown | -19% | -19% |

| Hit Rate | 73% | 62% |

| Cumulative Return | 512× | — |



\## Approach

\- \*\*Data\*\*: Daily OHLCV from Yahoo Finance (2017–2025)

\- \*\*Features\*\*: 35+ technical indicators including momentum, volatility, RSI, MACD, Bollinger Bands, cross-sectional ranks

\- \*\*Model\*\*: Ensemble of Logistic Regression + Random Forest + XGBoost (soft voting)

\- \*\*Signal\*\*: Predict P(positive next-week return) for each stock every Friday

\- \*\*Portfolio\*\*: Top-2 stocks by predicted probability, equal-weight, long-only

\- \*\*Rebalance\*\*: Weekly (every Friday close)

\- \*\*Transaction Costs\*\*: 0.1% entry + 0.1% exit



\## Validation

\- Train: 2017–2022

\- Test: 2023–2025

\- Walk-forward retraining (yearly)



\## Files

| File | Description |

|---|---|

| `QuantQuest\_Momentum\_Strategy.ipynb` | Main notebook |

| `backtest\_report.png` | 6-panel performance chart |

| `feature\_importance.png` | Top-20 feature importances |

| `weekly\_predictions.csv` | Weekly stock predictions and weights |

| `QuantQuest\_Report.pptx` | Presentation slides |



\## How to Run

```bash

pip install yfinance scikit-learn xgboost matplotlib seaborn pandas numpy jupyter

jupyter notebook QuantQuest\_Momentum\_Strategy.ipynb

```

Then click \*\*Cell → Run All\*\*



\## Key Design Choices

\- \*\*Skip-1-week momentum\*\*: avoids short-term reversal effect

\- \*\*Cross-sectional ranking\*\*: compares stocks against each other

\- \*\*Soft voting ensemble\*\*: blends probability estimates from 3 models

\- \*\*Look-ahead bias fix\*\*: signals shifted by 1 week before execution



\## Author

Keerthana G | E-Summit '26 | IIT Mandi

