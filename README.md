# Regime Alpha Long-Short Trading System

Research-to-execution workflow for systematic trading: regime signal modeling, no-lookahead backtesting, portfolio risk control, and Alpaca paper-trading guards.

This project started as a single-stock Hidden Markov Model regime classifier. The useful result came after reframing the signal as a cross-sectional long-short portfolio problem and controlling broad market exposure with a causal rolling CAPM net-beta cap.

## What I Built

- 30-minute U.S. equity OHLCV data pipeline.
- HMM-based regime labeling for Bull / Side / Bear states.
- Supervised meta-modeling using HMM posterior probabilities, transition priors, trend features, and volume features.
- Continuous position signal: `P(Bull) - P(Bear)`.
- 49-stock long-short portfolio construction.
- Causal rolling CAPM net-beta cap using prior daily closes only.
- No-lookahead backtest workflow with explicit train/test periods and shifted execution timing.
- Alpaca paper-trading prototype with dry-run mode, order reconciliation, data freshness checks, shortability checks, manifest validation, and guard logs.

## Key Result

Period: 2024-01-02 to 2026-05-21. Universe: 49 liquid U.S. equities. SPY is used as the market benchmark, not as a traded portfolio member.

These are gross research diagnostics, not deployable net performance.

| Portfolio | Net Beta Cap | Cumulative Return | Annualized Return | Sharpe, rf = 0% | Max Drawdown | Realized Beta vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| Equal-Weight Long-Short | None | +54.8% | +20.2% | 2.25 | -6.8% | -0.20 |
| Equal-Weight Long-Short | 0.25 | +54.5% | +20.1% | 3.41 | -2.4% | -0.10 |
| SPY Buy & Hold | n/a | +57.2% | +20.9% | 1.30 | -19.0% | 1.00 |

The beta-capped portfolio kept nearly the same return as the uncapped version while materially reducing drawdown and volatility. Its realized beta was close to market-neutral, so the result is better interpreted as residual alpha evidence than as a disguised SPY bet.

A more conservative risk-free-rate check gives lower, more realistic Sharpe estimates:

| One-Way Slippage | Sharpe, rf = 0% | Sharpe, rf = 4.5% |
|---:|---:|---:|
| 0 bp | 3.41 | 2.58 |
| 2 bp | 3.24 | 2.41 |
| 5 bp | 2.99 | 2.16 |
| 10 bp | 2.56 | 1.73 |

## Methodology

```text
30-minute regular-session OHLCV data
        |
        v
Rolling feature engineering
        |
        v
Gaussian HMM regime labeling
        |
        v
ADX / R2 classifiers + transition features + RVOL features
        |
        v
Logistic meta-model for next-regime probabilities
        |
        v
Continuous signal: P(Bull) - P(Bear)
        |
        v
49-stock long-short portfolio
        |
        v
Causal rolling CAPM net-beta cap
        |
        v
Backtest diagnostics / Alpaca paper-trading prototype
```

The first version applied the HMM strategy to individual stocks. It beat the Donchian + ADX/R2 trend baseline on 10/10 tested names, but beat buy-and-hold on only 3/10 names. That failure mode suggested the model was better understood as a regime-aware trend filter than as a standalone single-name alpha engine.

The portfolio version asks a different question:

> Can regime signals identify a basket of stronger-trending stocks to long and weaker or bear-regime stocks to short, while keeping broad market beta small?

## Backtest Hygiene

- Training data and test data are separated by explicit dates.
- Warm-up bars are included for rolling features but excluded from reported OOS statistics.
- Signal timing is shifted by one bar: information available at bar `t` is executed at `open[t+1]`.
- Rolling beta estimates use prior daily closes only.
- SPY is used as the benchmark for beta estimation, not as a traded hedge.
- The beta cap does not flip position signs or create new hedge positions.

## Repository Structure

```text
.
|-- strategy/
|   |-- HMM_strategy/
|   |   |-- classifiers/       # ADX and R2 regime classifiers
|   |   |-- features/          # rolling price, trend, scaler, and RVOL features
|   |   |-- meta_model/        # logistic meta-model wrapper
|   |   |-- position/          # probability-to-position sizing
|   |   |-- regime/            # HMM labeler, transition model, label smoother
|   |   |-- scripts/           # feature and regime verification utilities
|   |   |-- allocations.py     # portfolio allocation and net-beta cap logic
|   |   |-- config.py          # experiment and live-trading configuration
|   |   `-- strategy.py        # integrated HMM strategy
|   |-- donchian_adx_r2_B.py   # trend-following baseline and hybrid filter
|   `-- ma_cross.py            # moving-average baseline
|-- backtester/                # single-name and continuous-position backtest tools
|-- analysis/                  # OOS signal extraction and portfolio diagnostics
|-- data/                      # data fetch and 30-minute resampling scripts
|-- tests/                     # HMM, allocation, position sizing, and live-guard tests
|-- run_backtest_hmm.py
|-- run_all_backtests.py
`-- live_trade.py
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

## Data And Credentials

The repository intentionally excludes generated data and private artifacts:

- `.env`
- raw or resampled parquet/csv market data
- model caches such as `models/*.joblib`
- generated HTML reports
- logs and live-trading outputs

Create a local `.env` file for Alpaca paper trading:

```text
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

The live trading script connects with `paper=True`. By default, it runs in dry-run mode and prints intended orders without submitting them.

## What This Demonstrates

- Turning a weak single-name model result into a stronger portfolio-level hypothesis.
- Time-series ML implementation with explicit leakage controls.
- Portfolio risk control using rolling beta estimation and constrained allocation.
- Research engineering around reproducible diagnostics and test coverage.
- Practical execution concerns for paper trading, including dry-run behavior, freshness checks, order reconciliation, market-open guards, and shortability checks.

## Limitations

- The headline portfolio result is gross of full transaction costs, borrow fees, borrow constraints, financing costs, and market impact.
- The OOS window is limited and should be extended through more market regimes, especially 2020 and 2022.
- The universe is based on liquid large-cap equities; point-in-time membership and survivorship-bias handling should be documented more rigorously.
- CAPM residual alpha may contain other factor exposures; a multi-factor regression is a natural next step.
- Paper trading validates execution plumbing, not live alpha.

## More Detail

See [`docs/backtest-report.md`](docs/backtest-report.md) for the research path, diagnostic interpretation, and planned next steps.

## Tech Stack

Python, pandas, NumPy, pyarrow, scikit-learn, hmmlearn, joblib, Plotly, matplotlib, alpaca-py, pytest.

## Status

Research prototype. The main current result is a portfolio-level OOS diagnostic showing that regime signals became more informative after cross-sectional long-short aggregation and explicit net-beta control.
