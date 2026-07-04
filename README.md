# Regime Alpha Long-Short Trading System

HMM-based U.S. equity regime modeling, long-short portfolio construction, and Alpaca paper-trading research system.

This project started as a single-stock Hidden Markov Model regime classifier. The main finding emerged after scaling the signals into a 49-stock long-short portfolio and constraining market exposure with a causal rolling CAPM net-beta cap.

## Executive Summary

- Built an end-to-end quant research pipeline: 30-minute OHLCV data engineering, HMM regime labeling, supervised meta-modeling, no-lookahead backtesting, portfolio diagnostics, and Alpaca paper-trading execution.
- Found that the single-stock HMM strategy behaved more like a regime-aware trend filter than a standalone alpha engine: it beats buy-and-hold on only 3/10 names.
- Reframed the signal as a cross-sectional long-short portfolio problem: long stocks classified as Bull, short stocks classified as Bear, and size each position by `P(Bull) - P(Bear)`.
- Added a causal rolling CAPM net-beta cap, `abs(sum(weight_i * beta_i)) <= 0.25`, to reduce broad market exposure without flipping signs or introducing hedge positions.
- In the 49-stock gross OOS diagnostic, the beta-capped portfolio produced `+54.5%` cumulative return, `+20.1%` annualized return, `3.41` Sharpe, and `-2.4%` max drawdown.
- The same beta-capped portfolio had realized market beta `-0.10` vs SPY and CAPM R2 `0.08`, suggesting the return was not primarily explained by linear market exposure.

## Key Result

Period: 2024-01-02 to 2026-05-21. Universe: 49 liquid U.S. equities. SPY is used as the market benchmark, not as a traded portfolio member. Results below are gross research diagnostics.

| Portfolio | Net Beta Cap | Cumulative Return | Annualized Return | Sharpe | Max Drawdown | Realized Beta vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| Equal-Weight Long-Short | None | +54.8% | +20.2% | 2.25 | -6.8% | -0.20 |
| Equal-Weight Long-Short | 0.25 | +54.5% | +20.1% | 3.41 | -2.4% | -0.10 |
| SPY Buy & Hold | n/a | +57.2% | +20.9% | 1.30 | -19.0% | 1.00 |

The beta-capped portfolio kept nearly the same return as the uncapped version while materially improving drawdown and volatility. Its realized beta was close to market-neutral, so the result is better interpreted as residual alpha evidence than as a disguised SPY bet.

* Sharpe Ratio is calcualted without risk free asset return. Risk free return is set to be 0, so Sharpe Ratio = Return/Volatility. Sharpe Ratio calculated with risk free return will be provided in the other section of this document. 

## Why This Looks Like Alpha, Not Market Beta

The portfolio dashboard includes a single-factor CAPM regression against SPY daily returns. For the beta-capped portfolio:

| CAPM Diagnostic | Equal-Weight Long-Short | Beta-Capped Long-Short |
|---|---:|---:|
| Realized beta | -0.20 | -0.10 |
| Annualized alpha | +22.9% | +20.5% |
| Correlation vs SPY | -0.38 | -0.29 |
| R2 vs SPY | 0.15 | 0.08 |
| Market beta contribution | -9.9% | -4.9% |
| Residual alpha contribution | +54.4% | +48.7% |

Because SPY rose strongly over the same period, a realized beta of `-0.10` would not explain the portfolio's positive return. In this single-factor decomposition, the estimated market contribution was negative while the residual component was strongly positive. This does not prove pure stock-selection alpha, because other factor exposures may be embedded in the residual, but it is strong evidence that the result was not driven by broad market beta.

## Research Path

The first version applied HMM regime classification to individual stocks. That produced mixed results:

- The best HMM-family variant beat the standalone Donchian + ADX/R2 trend baseline on 10/10 tested names.
- It beat buy-and-hold on only 3/10 names.
- It struggled against high-beta momentum winners such as NVDA, MU, and SOXL, where holding the asset through the bull trend was hard to beat.

That failure mode was useful. It suggested the HMM was detecting trend regimes, but the single-name framing was asking the wrong question. Instead of using the model to decide whether one stock should beat its own buy-and-hold path, I used the regime scores to construct a diversified long-short book across many stocks.

The portfolio version asks a different question:

> Can regime signals identify a basket of stronger-trending stocks to long and weaker or bear-regime stocks to short, while keeping broad market beta small?

That reframing is where the performance improved most.

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

The HMM states are mapped by their realized regime characteristics:

- `Bull`: highest rolling cumulative return state, usually stronger trend quality.
- `Side`: middle state, usually lower trend strength.
- `Bear`: lowest rolling cumulative return state.

The meta-model can use HMM posterior probabilities, transition priors, ADX/R2 regime classifier probabilities, rolling price-window features, and time-of-day adjusted relative-volume features.

## Portfolio Construction

For each symbol, the model produces a continuous signal in `[-1.0, +1.0]`:

```text
signal_i = P_i(Bull) - P_i(Bear)
raw_weight_i = allocation_i * signal_i
```

The current diagnostic uses equal risk budget per stock before signal scaling. After raw weights are created, a no-lookahead rolling beta model estimates each stock's CAPM beta against SPY using prior daily closes only. If the portfolio's estimated net beta exceeds the configured cap, the allocator scales down only the side of the book that creates excess beta.

Important implementation constraints:

- No future returns are used in beta estimation.
- The beta cap does not flip position signs.
- The beta cap does not create new hedge positions.
- SPY is used as the benchmark for beta estimation, not as a traded asset.

Core implementation: `strategy/HMM_strategy/allocations.py`.

## Backtest Design

- Training data and test data are separated by explicit dates.
- Warm-up bars are included for rolling features but excluded from reported OOS statistics.
- Signal timing is shifted by one bar: information available at bar `t` is executed at `open[t+1]`.
- The portfolio dashboard uses OOS signals from `analysis/oos_signals.parquet`.
- Transaction costs, market impact, borrow fees, and borrow availability are not fully modeled in the headline result.
- Initial backtest design was done without the consideration of market fee and slippage, as alpaca does not impose market fee and the 49 equities in the portfolio are traded in a huge trading volume, so the size of slippage is not to be too large. However, the results with slippage are also provided. 

Slippage sensitivity for the beta-capped portfolio:

| One-Way Slippage | Cumulative Return | Annualized Return | Sharpe | Max Drawdown |
|---:|---:|---:|---:|---:|
| 0 bp | +54.5% | +20.1% | 3.41 | -2.4% |
| 2 bp | +51.1% | +19.0% | 3.24 | -2.4% |
| 5 bp | +46.2% | +17.3% | 2.99 | -2.5% |
| 10 bp | +38.4% | +14.6% | 2.56 | -2.6% |

Sharpe Sensitivity, Slippage and Risk-Free Rate:

The BetaCap 0.25 portfolio remained robust under more conservative assumptions for both transaction slippage and risk-free rate.  
Even with a 4.5% annual risk-free rate and 10bp one-way slippage, the portfolio still produced a Sharpe ratio of `1.73`.

| One-Way Slippage | rf = 0% (current code) | rf = 4.0% | rf = 4.5% | rf = 5.0% |
|---:|---:|---:|---:|---:|
| 0bp | 3.41 | 2.67 | **2.58** | 2.49 |
| 2bp | 3.24 | 2.50 | **2.41** | 2.32 |
| 5bp | 2.99 | 2.25 | **2.16** | 2.06 |
| 10bp | 2.56 | 1.82 | **1.73** | 1.63 |

This sensitivity check suggests that the result is not dependent on the zero-risk-free-rate assumption used in the current backtest code. The strategy's risk-adjusted performance remains positive after applying both realistic cash-rate assumptions and moderate transaction slippage.

## Single-Name Backtest Summary

Period: 2025-01-01 to 2026-05-22, 30-minute bars. The HMM column reports the best Sharpe variant among four HMM configurations.

| Result | Count |
|---|---:|
| HMM-family variant beat Donchian + ADX/R2 baseline | 10 / 10 |
| HMM-family variant beat buy-and-hold | 3 / 10 |

This is why the README emphasizes portfolio construction rather than only single-name model accuracy. The alpha became more visible after the signal was diversified across a long-short book.

## Interactive Dashboard

The portfolio dashboard is generated by:

```bash
python analysis/portfolio_dashboard.py
```

Local output:

```text
results/portfolio_dashboard.html
```

The dashboard contains:

- Equal-weight vs beta-capped equity curves.
- SPY buy-and-hold comparison.
- Slippage sensitivity tables.
- CAPM regression decomposition.
- Hoverable daily long/short holdings and net beta diagnostics.

For a public GitHub portfolio, I would keep the README concise and publish the full dashboard separately through one of these routes:

- GitHub Pages HTML dashboard.
- A `docs/backtest-report.md` research write-up with selected charts.
- Static screenshots in `docs/assets/` plus the generated HTML as a downloadable artifact.

## Repository Structure

```text
.
├── strategy/
│   ├── HMM_strategy/
│   │   ├── classifiers/       # ADX and R2 regime classifiers
│   │   ├── features/          # rolling price, trend, scaler, and RVOL features
│   │   ├── meta_model/        # logistic meta-model wrapper
│   │   ├── position/          # probability-to-position sizing
│   │   ├── regime/            # HMM labeler, transition model, label smoother
│   │   ├── scripts/           # feature and regime verification utilities
│   │   ├── allocations.py     # portfolio allocation and net-beta cap logic
│   │   ├── config.py          # experiment and live-trading configuration
│   │   └── strategy.py        # integrated HMM strategy
│   ├── donchian_adx_r2_B.py   # trend-following baseline and hybrid filter
│   └── ma_cross.py            # moving-average baseline
├── backtester/                # single-name and continuous-position backtest tools
├── analysis/                  # OOS signal extraction and portfolio diagnostics
├── data/                      # data fetch and 30-minute resampling scripts
├── tests/                     # HMM, allocation, and live-guard tests
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

## What This Project Demonstrates

- Quant research process: turning a weak single-name result into a stronger portfolio-level hypothesis.
- Time-series ML implementation with explicit leakage controls.
- Portfolio risk control using rolling beta estimation and constrained allocation.
- Research engineering around generated data, reproducible diagnostics, and test coverage.
- Practical execution concerns for paper trading, including dry-run behavior and order safety checks.

## Limitations And Next Steps

- The headline portfolio result is gross of full transaction costs, borrow fees, borrow constraints, financing costs, and market impact.
- The OOS window is still limited and should be extended through 2020 and 2022 bear-market regimes.
- The universe is based on liquid large-cap equities; point-in-time membership and survivorship-bias handling should be documented more rigorously for final publication.
- CAPM residual alpha may include other factor exposures; a multi-factor regression would be a natural next step.
- Paper trading validates execution plumbing, not live alpha.

## Tech Stack

- Python
- pandas, NumPy, pyarrow
- scikit-learn, hmmlearn, joblib
- Plotly, matplotlib
- alpaca-py
- pytest

## Status

Research prototype. The main current result is a portfolio-level OOS diagnostic showing that regime signals became more informative after cross-sectional long-short aggregation and explicit net-beta control.
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
