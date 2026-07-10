# Backtest Report

This note provides additional context for the headline README result. It is not a live trading performance report.

## Research Path

The project began with a single-name HMM regime classifier. The original hypothesis was simple: if hidden market regimes can be inferred from 30-minute price and volume data, the model may help decide whether a stock should be long, short, or flat.

The single-name tests were useful, but not strong enough as a standalone alpha result:

- The best HMM-family variant beat the Donchian + ADX/R2 trend baseline on 10/10 tested names.
- It beat buy-and-hold on only 3/10 tested names.
- It struggled most against strong directional momentum winners, where simply holding the asset through a bull trend was difficult to beat.

That failure mode changed the research question. The model looked more like a regime-aware trend filter than a pure single-name prediction engine. I therefore reframed the problem from:

> Can the model beat buy-and-hold on one stock?

to:

> Can the model identify relatively stronger and weaker regime states across many stocks, then convert that ranking into a beta-controlled long-short book?

## Portfolio Construction

For each symbol, the model produces a continuous signal:

```text
signal_i = P_i(Bull) - P_i(Bear)
raw_weight_i = allocation_i * signal_i
```

The diagnostic portfolio uses equal risk budget per stock before signal scaling. After raw weights are created, a rolling beta model estimates each stock's CAPM beta against SPY using prior daily closes only. If estimated portfolio net beta exceeds the configured cap, the allocator scales down only the side of the book that creates excess beta.

Implementation constraints:

- No future returns are used in beta estimation.
- The beta cap does not flip position signs.
- The beta cap does not create new hedge positions.
- SPY is a benchmark and beta reference, not a traded asset.

Core implementation: `strategy/HMM_strategy/allocations.py`.

## Portfolio Diagnostic

Period: 2024-01-02 to 2026-05-21. Universe: 49 liquid U.S. equities.

| Portfolio | Net Beta Cap | Cumulative Return | Annualized Return | Sharpe, rf = 0% | Max Drawdown | Realized Beta vs SPY |
|---|---:|---:|---:|---:|---:|---:|
| Equal-Weight Long-Short | None | +54.8% | +20.2% | 2.25 | -6.8% | -0.20 |
| Equal-Weight Long-Short | 0.25 | +54.5% | +20.1% | 3.41 | -2.4% | -0.10 |
| SPY Buy & Hold | n/a | +57.2% | +20.9% | 1.30 | -19.0% | 1.00 |

The beta cap reduced drawdown from -6.8% to -2.4% while keeping cumulative return almost unchanged. Since SPY rose strongly over the same period, the beta-capped portfolio's realized beta of -0.10 does not explain its positive return.

## CAPM Decomposition

| CAPM Diagnostic | Equal-Weight Long-Short | Beta-Capped Long-Short |
|---|---:|---:|
| Realized beta | -0.20 | -0.10 |
| Annualized alpha | +22.9% | +20.5% |
| Correlation vs SPY | -0.38 | -0.29 |
| R2 vs SPY | 0.15 | 0.08 |
| Market beta contribution | -9.9% | -4.9% |
| Residual alpha contribution | +54.4% | +48.7% |

This does not prove pure stock-selection alpha. Other factor exposures may be embedded in the residual. It does suggest that the result was not primarily driven by broad linear market beta.

## Slippage And Risk-Free-Rate Sensitivity

| One-Way Slippage | Cumulative Return | Annualized Return | Sharpe, rf = 0% | Sharpe, rf = 4.5% | Max Drawdown |
|---:|---:|---:|---:|---:|---:|
| 0 bp | +54.5% | +20.1% | 3.41 | 2.58 | -2.4% |
| 2 bp | +51.1% | +19.0% | 3.24 | 2.41 | -2.4% |
| 5 bp | +46.2% | +17.3% | 2.99 | 2.16 | -2.5% |
| 10 bp | +38.4% | +14.6% | 2.56 | 1.73 | -2.6% |

The strategy remains positive under moderate one-way slippage and a 4.5% annual risk-free-rate assumption, but this is still a simplified diagnostic.

## Execution Prototype

The project includes an Alpaca paper-trading prototype in `live_trade.py`. It is designed to validate research-to-execution plumbing, not to prove live alpha.

The live script includes:

- dry-run mode by default
- paper account connection only
- completed-bar timing logic
- data freshness checks
- market-open guards
- shortability checks
- manifest freshness and approval checks
- order reconciliation
- guard and beta logs

Related tests are in `tests/test_live_trade_guards.py`.

## Limitations

- Full transaction costs, borrow fees, borrow availability, financing costs, and market impact are not modeled in the headline result.
- The OOS period is limited and should be extended to more regimes.
- The universe is based on liquid large-cap equities; point-in-time membership and survivorship-bias handling need stricter documentation.
- CAPM residual alpha can contain other factor exposures.
- The paper-trading prototype validates execution plumbing, not live alpha.

## Next Steps

- Extend walk-forward tests through 2020 and 2022.
- Add multi-factor regression diagnostics.
- Improve borrow/financing cost assumptions for the short book.
- Add point-in-time universe documentation.
- Publish generated dashboard screenshots under `docs/assets/` or GitHub Pages.
