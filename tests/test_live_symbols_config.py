import unittest
from pathlib import Path

import pandas as pd

from strategy.HMM_strategy import config


ROOT = Path(__file__).resolve().parents[1]


class LiveSymbolsConfigTest(unittest.TestCase):
    def test_live_symbols_match_candidate_universe_without_gev_or_spy(self):
        candidate = pd.read_csv(ROOT / "plans" / "candidate_universe_v2.csv")
        expected = [symbol for symbol in candidate["sym"].tolist() if symbol != "GEV"]

        self.assertEqual(config.LIVE_SYMBOLS, expected)
        self.assertEqual(len(config.LIVE_SYMBOLS), 49)
        self.assertNotIn("GEV", config.LIVE_SYMBOLS)
        self.assertNotIn("SPY", config.LIVE_SYMBOLS)

    def test_live_symbols_and_benchmark_have_30min_parquet(self):
        data_dir = ROOT / config.LIVE_DATA_DIR
        symbols = [*config.LIVE_SYMBOLS, config.LIVE_BETA_BENCHMARK_SYMBOL]

        missing = [
            symbol
            for symbol in symbols
            if not list(data_dir.glob(f"{symbol}_*_30min.parquet"))
        ]

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
