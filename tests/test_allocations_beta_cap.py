import unittest

import numpy as np
import pandas as pd

from strategy.HMM_strategy.allocations import (
    apply_net_beta_cap,
    calculate_net_beta,
    estimate_capm_betas_from_daily_closes,
)


class NetBetaCapSmokeTest(unittest.TestCase):
    def test_caps_net_beta_without_flipping_signs(self):
        raw = {"A": 0.40, "B": 0.10, "C": -0.05}
        betas = {"A": 1.00, "B": 0.50, "C": -1.00}

        result = apply_net_beta_cap(raw, betas, cap=0.25)

        self.assertTrue(result.capped)
        self.assertLessEqual(abs(result.adjusted_net_beta), 0.25 + 1e-12)
        for symbol, raw_weight in raw.items():
            adjusted = result.adjusted_weights[symbol]
            self.assertLessEqual(abs(adjusted), abs(raw_weight) + 1e-12)
            self.assertGreaterEqual(adjusted * raw_weight, 0.0)

    def test_leaves_weights_unchanged_when_inside_cap(self):
        raw = {"A": 0.10, "B": -0.05}
        betas = {"A": 1.00, "B": 0.50}

        result = apply_net_beta_cap(raw, betas, cap=0.25)

        self.assertFalse(result.capped)
        self.assertEqual(result.adjusted_weights, raw)
        self.assertEqual(result.adjusted_net_beta, calculate_net_beta(raw, betas))

    def test_missing_beta_is_excluded_without_changing_weight(self):
        raw = {"A": 0.40, "MISSING": 0.20}
        betas = {"A": 1.00}

        result = apply_net_beta_cap(raw, betas, cap=0.25)

        self.assertIn("MISSING", result.missing_beta_symbols)
        self.assertEqual(result.adjusted_weights["MISSING"], raw["MISSING"])
        self.assertLessEqual(abs(result.adjusted_net_beta), 0.25 + 1e-12)

    def test_beta_estimation_excludes_as_of_date_return(self):
        close_daily = pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0, 103.0, 200.0],
                "A": [100.0, 102.0, 104.0, 106.0, 10.0],
            },
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )

        beta_map, missing = estimate_capm_betas_from_daily_closes(
            close_daily,
            symbols=["A"],
            benchmark_symbol="SPY",
            as_of_date="2024-01-05",
            lookback_days=10,
            min_obs=2,
        )
        beta_with_future, _ = estimate_capm_betas_from_daily_closes(
            close_daily,
            symbols=["A"],
            benchmark_symbol="SPY",
            as_of_date="2024-01-06",
            lookback_days=10,
            min_obs=2,
        )

        returns = close_daily.pct_change(fill_method=None)
        hist = returns.loc[returns.index < pd.Timestamp("2024-01-05")]
        expected = np.cov(hist["A"].dropna(), hist["SPY"].dropna())[0, 1]
        expected /= np.var(hist["SPY"].dropna())

        self.assertEqual(missing, {})
        self.assertAlmostEqual(beta_map["A"], expected)
        self.assertNotAlmostEqual(beta_map["A"], beta_with_future["A"])


if __name__ == "__main__":
    unittest.main()
