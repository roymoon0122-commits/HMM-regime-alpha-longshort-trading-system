# Regime Alpha Trading System

HMM-based U.S. equity regime detection, backtesting, and Alpaca paper-trading research system.

This project explores whether market-regime classification can improve trend-following strategies by adjusting exposure across bull, sideways, and bear regimes. The implementation includes feature engineering, Gaussian HMM regime labeling, supervised meta-modeling, position sizing, backtesting, portfolio diagnostics, and a paper-trading execution prototype.


## Highlights

- Builds 30-minute regular-session OHLCV datasets for U.S. equities.
- Computes rolling regime features such as cumulative return, volatility, ADX, R-squared trend quality, slope, drawdown, candle direction, and relative-volume features.
- Trains a Gaussian HMM with K-means initialization, random restarts, transition priors, and automatic state-to-regime mapping.
- Uses a logistic meta-model and optional retrospective label smoothing to estimate next-regime probabilities.
- Converts regime probabilities into continuous target exposure in `[-1.0, +1.0]`.
- Runs look-ahead-safe backtests where signals from bar `t` are executed at the next bar open.
- Compares HMM variants against Donchian + ADX/R², MA crossover, and buy-and-hold baselines.
- Includes an Alpaca paper-trading loop with dry-run mode, order safety checks, and GitHub Actions scheduling.

## Research Question

Can a regime-aware trend system improve drawdown control or risk-adjusted returns compared with simpler trend-following baselines?

The current answer is mixed:

- On the 10 single-name backtest set, the best HMM-family variant beat the standalone Donchian + ADX/R² baseline on 10 of 10 names, but beat buy-and-hold on only 3 of 10 names.
- On a separate 49-stock equal-weight gross portfolio diagnostic, the signal set showed strong risk-adjusted performance before transaction costs.
- With long-short portfolio, the risk exposure of whole portfolio to the market has significantly dropped(MDD less than 3%). 

## Methodology

```text
30-minute OHLCV data
        |
        v
Rolling feature engineering
        |
        v
Rolling normalization + HMM regime labeling
        |
        v
ADX / R-squared classifiers + transition features + RVOL features
        |
        v
Logistic meta-model for next-regime probabilities
        |
        v
Position sizing: P(Bull) - P(Bear)
        |
        v
Long-Short Portfolio with |net Beta(of CAPM)| < 0.25
        |
        v
Backtest engine / Alpaca paper-trading execution
```

The HMM labels regimes as:

- `Bull`: highest average rolling cumulative return, high ADX
- `Side`: middle state, low ADX
- `Bear`: lowest average rolling cumulative return state, high ADX

The meta-model can optionally use:

- HMM posterior probabilities
- HMM transition prior probabilities
- ADX and R² regime classifier probabilities
- rolling price-window features
- relative-volume features based on time-of-day adjusted RVOL

## Backtest Design

The main backtest script separates training and out-of-sample evaluation:

- Training data: `train_start` through `train_end`
- Test data: `test_start` through `test_end`
- Warm-up bars are included before the OOS start for rolling scalers and indicators, but excluded from reported statistics.
- Signal timing is shifted by one bar: information available at bar `t` is traded at `open[t+1]`.
- The default fee/slippage assumption in `run_backtest_hmm.py` is `0.0003` per rebalance.

## Results Snapshot

### Single-Name OOS Backtests

Period: 2025-01-01 to 2026-05-22, 30-minute bars. The HMM column reports the best Sharpe variant among four HMM configurations.

| Symbol | Best HMM Variant | HMM CAGR / Sharpe | Donchian CAGR / Sharpe | Buy & Hold CAGR / Sharpe |
|---|---|---:|---:|---:|
| AAPL | HMM + Smooth + Donchian | +29.8% / 1.21 | +19.6% / 1.20 | +22.9% / 0.89 |
| AMZN | HMM + Smooth + Donchian | +0.6% / 0.15 | -32.4% / -1.85 | +27.9% / 0.91 |
| GOOGL | HMM, no smoothing | +37.8% / 1.55 | -11.5% / -0.68 | +54.9% / 1.54 |
| META | HMM + Smooth | +45.7% / 1.30 | -24.0% / -1.06 | +26.9% / 0.81 |
| MSFT | HMM + Smooth | +12.4% / 0.74 | -10.4% / -0.58 | +5.6% / 0.35 |
| MU | HMM + Smooth | +52.1% / 1.18 | -7.5% / 0.02 | +152.7% / 1.84 |
| NVDA | HMM, no smoothing + Donchian | +42.3% / 1.14 | +17.0% / 0.73 | +89.5% / 1.59 |
| SOXL | HMM + Smooth + Donchian | +26.3% / 0.68 | +4.6% / 0.40 | +114.1% / 1.25 |
| SPY | HMM + Smooth + Donchian | +8.6% / 0.68 | -6.2% / -0.57 | +20.9% / 1.30 |
| TSLA | HMM, no smoothing | +10.1% / 0.44 | -19.9% / -0.43 | +24.3% / 0.67 |

Summary:

- Best HMM-family variant beat buy-and-hold on 3 / 10 names.
- Best HMM-family variant beat the standalone Donchian + ADX/R² baseline on 10 / 10 names.
- The model tended to help more on names where regime shifts and drawdown control mattered, and less on strong directional winners.
- The model seems to follow up the market trend well but does not seem to generate meaningful "alpha".
---> The Idea: As the model follows up the makret trend well, by constructing a long-short portfolio which longs on the stocks classified to be "Bull" and shorts of the stocks classified to be "Bear", we can hedge the market risk while benefiting by "alpha".
  
### Portfolio-Level Diagnostic

Source: `analysis/oos_signals.parquet` and `analysis/portfolio_decomp.py`

This is an equal-weight gross signal diagnostic over 49 U.S. equities, against SPY as the benchmark reference. Transaction costs, borrow constraints, and market impact are not included in this diagnostic -- as alpaca does not impose any fee and these 49 equities have very rich amount of transactions no sgnificiant amount of slippage expected. Backtesting results with slippage and fee will be added.

| Metric | Value |
|---|---:|
| Period | 2024-01-02 to 2026-05-21 |
| Universe | 49 stocks |
| Cumulative return | +54.8% |
| Annualized return | +20.2% |
| Annualized volatility | 8.3% |
| Daily Sharpe | 2.25 |
| Max drawdown | -6.8% |
| Market beta vs SPY | -0.20 |
| Residual alpha estimate | +22.7% / year |
| Market R² | 14.3% |
| Average gross exposure | 59.8% |

### Net-Beta-Capped Portfolio

Source: `analysis/portfolio_dashboard.py`

The expanded portfolio experiment also tests a causal rolling CAPM beta estimate and a net beta cap. The cap scales down only the side of the book that causes excess net beta; it does not flip position signs or create new positions. The beta estimate uses prior daily closes only. 

Period: 2024-01-02 to 2026-05-21. Universe: 49 stocks. Gross OOS diagnostic.

| Portfolio | Net Beta Cap | Cumulative Return | Annualized Return | Sharpe | MDD | Realized Beta vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| Equal-Weight | None | +54.8% | +20.2% | 2.25 | -6.8% | -0.20 |
| Equal-Weight BetaCap | 0.25 | +54.5% | +20.1% | 3.41 | -2.4% | -0.10 |
| SPY Buy & Hold | n/a | +57.2% | +20.9% | 1.30 | -19.0% | 1.00 |

Slippage sensitivity for `Equal-Weight BetaCap 0.25`:

| One-Way Slippage | Cumulative Return | Annualized Return | Sharpe | MDD |
|---:|---:|---:|---:|---:|
| 0 bp | +54.5% | +20.1% | 3.41 | -2.4% |
| 2 bp | +51.1% | +19.0% | 3.24 | -2.4% |
| 5 bp | +46.2% | +17.3% | 2.99 | -2.5% |
| 10 bp | +38.4% | +14.6% | 2.56 | -2.6% |

This result should be read as a research diagnostic, not as deployable net performance. 
The next research step is longer walk-forward tests including 2020 and 2022. Especially, the backtesting was not done on the market of long time bear the robustness of the alpha is not to be trusted fully. However, the OOS period of the backtesting does include the crash of market (2025.02 - 2025.04, 2026.03) and the return of the portfolio did not falter.
Also, the shorting assumptions are simplified, real borrow availability and financing costs are not modeled. If possible, those limitations on shorting must be included and backtested. This is another research subject to be done. 

## Repository Structure

```text
.
├── strategy/
│   ├── HMM_strategy/
│   │   ├── classifiers/       # ADX and R-squared regime classifiers
│   │   ├── features/          # rolling price, trend, scaler, and RVOL features
│   │   ├── meta_model/        # logistic meta-model wrapper
│   │   ├── position/          # probability-to-position sizing
│   │   ├── regime/            # HMM labeler, transition model, label smoother
│   │   ├── scripts/           # feature and regime verification utilities
│   │   ├── allocations.py     # live portfolio allocation placeholder
│   │   ├── config.py          # central experiment configuration
│   │   └── strategy.py        # integrated HMM strategy
│   ├── donchian_adx_r2_B.py   # trend-following baseline and hybrid filter
│   └── ma_cross.py            # moving-average baseline
├── backtester/
│   ├── engine.py              # integer-position backtester
│   ├── backtester_hmm.py      # continuous-position HMM backtester
│   ├── portfolio.py
│   ├── portfolio_continuous.py
│   └── visualizer_run_backtest_hmm.py
├── analysis/
│   ├── extract_oos_signals.py
│   ├── portfolio_decomp.py
│   └── beta_sweep_shortleg.py
├── data/
│   ├── 1_fetch_minute_bars.py
│   └── 2_resample_bars.py
├── tests/
├── run_backtest_hmm.py
├── run_all_backtests.py
└── live_trade.py
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run one HMM backtest:

```bash
python run_backtest_hmm.py \
  --csv-path data/30min/NVDA_20210101_20260527_30min.parquet \
  --hmm-cache models/hmm_nvda.joblib \
  --output-html results/backtest_hmm_NVDA.html
```

Run selected symbols:

```bash
python run_all_backtests.py --symbols AAPL GOOGL META MSFT NVDA TSLA
```

Run Alpaca paper-trading in dry-run mode:

```bash
python live_trade.py --once
```

Submit paper orders:

```bash
python live_trade.py --once --execute
```

## Alpaca Setup

Create a local `.env` file:

```text
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

The live trading script connects with `paper=True`. By default, it runs in dry-run mode and prints intended orders without submitting them.

## Data And Artifacts

The project expects resampled 30-minute parquet files under `data/30min/`. Generated data and artifacts should generally stay out of the portfolio repository:

- `.env`
- local virtual environments
- raw or large parquet/csv data
- model caches such as `models/*.joblib`
- generated HTML reports
- logs and live trading outputs
- Python cache files

For a public portfolio version, it is better to include small sample outputs or screenshots, then link to a separate write-up for the full backtest interpretation.

## Why Backtest Results Should Be A Separate Write-Up

README files should make the project understandable quickly. Detailed backtest interpretation is better as a separate article or `docs/backtest-report.md`, because it can discuss:

- the original hypothesis
- data period and universe construction
- train/test split and leakage controls
- parameter choices
- strategy variants
- benchmark selection
- where the model wins and fails
- sensitivity to fees, slippage, and shorting assumptions
- next research steps

The README should keep only the headline results and reproduction commands.

## Limitations

- The single-name result set is small and covers a limited market period.
- The portfolio-level diagnostic is gross of transaction costs and market impact.
- Shorting assumptions are simplified; real borrow availability and financing costs are not modeled.
- Buy-and-hold can outperform regime-filtered systems during strong momentum markets.
- HMM state labels are unsupervised and can be sensitive to feature choices, scaling, and covariance assumptions.
- Paper trading validates execution plumbing, not live alpha.

## Tech Stack

- Python
- pandas, NumPy, pyarrow
- scikit-learn, hmmlearn, joblib
- Plotly, matplotlib
- alpaca-py
- pytest

## Status

Research prototype. The current implementation is useful for studying regime-aware allocation logic, backtest hygiene, and paper-trading execution. The main planned improvements are portfolio construction, realistic cost modeling, walk-forward validation, and cleaner public research documentation.
