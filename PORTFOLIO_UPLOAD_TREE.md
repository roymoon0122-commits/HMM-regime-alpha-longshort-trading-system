# Portfolio Upload Tree

This document lists the files intended for the public portfolio repository.

## Include

```text
Stock-trader/
├── README.md
├── requirements.txt
├── .gitignore
├── run_backtest_hmm.py
├── run_all_backtests.py
├── live_trade.py
├── data/
│   ├── 1_fetch_minute_bars.py
│   └── 2_resample_bars.py
├── lib/
│   └── sp500_universe.py
├── backtester/
├── strategy/
├── analysis/
│   ├── extract_oos_signals.py
│   ├── extract_walkforward_signals.py
│   ├── portfolio_decomp.py
│   ├── walkforward_decomp.py
│   ├── beta_sweep_shortleg.py
│   ├── short_mechanism.py
│   └── portfolio_dashboard.py
├── plans/
│   └── candidate_universe_v2.csv
└── tests/
```

## Exclude

```text
.env
.venv/
venv/
__pycache__/
.pytest_cache/
.DS_Store
data/**/*.parquet
models/*.joblib
results/
logs/
analysis/*.parquet
analysis/*.csv
analysis/sig_parts/
analysis/sig_parts_wf/
*.html
*.png
```

## Notes

- Data and model files are generated artifacts and should not be committed.
- Backtest and portfolio analysis scripts are included so the research can be reproduced after data is regenerated locally.
- Alpaca credentials must be provided through a local `.env` file or GitHub Actions secrets, never through Git.
