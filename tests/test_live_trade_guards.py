import unittest
import importlib.util
from types import SimpleNamespace

import pandas as pd

ALPACA_AVAILABLE = importlib.util.find_spec("alpaca") is not None
if ALPACA_AVAILABLE:
    import live_trade
else:
    live_trade = None


@unittest.skipUnless(ALPACA_AVAILABLE, "alpaca-py is not installed")
class LiveTradeGuardTests(unittest.TestCase):
    def test_expected_completed_bar_waits_for_first_30m_bar(self):
        pre_first = pd.Timestamp("2026-07-01 09:45", tz=live_trade.MARKET_TZ)
        after_first = pd.Timestamp("2026-07-01 10:01", tz=live_trade.MARKET_TZ)

        self.assertIsNone(live_trade._expected_completed_bar_start(pre_first))
        self.assertEqual(
            live_trade._expected_completed_bar_start(after_first),
            pd.Timestamp("2026-07-01 09:30"),
        )

    def test_data_freshness_blocks_fetch_failure(self):
        now = pd.Timestamp("2026-07-01 10:31", tz=live_trade.MARKET_TZ)
        status = live_trade.LiveDataStatus(
            symbol="AAPL",
            fetch_failed=True,
            fetch_error="timeout",
            last_bar=pd.Timestamp("2026-07-01 10:00"),
        )

        ok, reason = live_trade._check_data_freshness(
            {"AAPL": status}, ["AAPL"], now)

        self.assertFalse(ok)
        self.assertIn("latest Alpaca fetch failed", reason)

    def test_data_freshness_accepts_expected_completed_bar(self):
        now = pd.Timestamp("2026-07-01 10:31", tz=live_trade.MARKET_TZ)
        status = live_trade.LiveDataStatus(
            symbol="AAPL",
            last_bar=pd.Timestamp("2026-07-01 10:00"),
        )

        ok, reason = live_trade._check_data_freshness(
            {"AAPL": status}, ["AAPL"], now)

        self.assertTrue(ok, reason)

    def test_cancel_open_orders_returns_false_on_timeout(self):
        order = SimpleNamespace(id="order-1")

        class FakeTradingClient:
            def get_orders(self, filter):
                return [order]

            def cancel_order_by_id(self, order_id):
                return None

        ok = live_trade.cancel_open_orders_for_symbols(
            FakeTradingClient(), ["AAPL"], timeout_sec=0.01, poll_sec=0.001)

        self.assertFalse(ok)

    def test_shortability_checks_only_new_or_increased_shorts(self):
        checked = []

        class FakeTradingClient:
            def get_asset(self, symbol):
                checked.append(symbol)
                return SimpleNamespace(
                    tradable=True,
                    shortable=True,
                    easy_to_borrow=True,
                )

        targets = {"AAPL": -10, "MSFT": -5, "TSLA": 5}
        positions = {"AAPL": 0, "MSFT": -20, "TSLA": 10}

        ok, reason = live_trade._check_shortability(
            FakeTradingClient(), targets, positions)

        self.assertTrue(ok, reason)
        self.assertEqual(checked, ["AAPL"])

    def test_market_submit_guard_blocks_closed_market(self):
        class FakeTradingClient:
            def get_clock(self):
                return SimpleNamespace(
                    is_open=False,
                    timestamp=pd.Timestamp(
                        "2026-07-01 08:00", tz=live_trade.MARKET_TZ),
                    next_close=pd.Timestamp(
                        "2026-07-01 16:00", tz=live_trade.MARKET_TZ),
                )

        ok, reason = live_trade._check_market_open_for_submit(
            FakeTradingClient())

        self.assertFalse(ok)
        self.assertIn("market closed", reason)

    def test_manifest_freshness_requires_previous_weekday_artifacts(self):
        manifest = {
            "schema_version": 1,
            "approved": True,
            "trained_through_date": "2026-06-30",
            "beta_asof_date": "2026-06-30",
            "universe": list(live_trade.SYMBOLS),
            "strategies": {sym: f"/tmp/{sym}.joblib" for sym in live_trade.SYMBOLS},
            "beta_by_symbol": {sym: 1.0 for sym in live_trade.SYMBOLS},
        }

        ok, reason = live_trade._check_manifest_freshness(
            manifest, pd.Timestamp("2026-07-01").date())

        self.assertTrue(ok, reason)

        manifest["beta_asof_date"] = "2026-06-29"
        ok, reason = live_trade._check_manifest_freshness(
            manifest, pd.Timestamp("2026-07-01").date())

        self.assertFalse(ok)
        self.assertIn("beta_asof_date", reason)

    def test_manifest_freshness_requires_approved_complete_manifest(self):
        manifest = {
            "schema_version": 1,
            "approved": False,
            "trained_through_date": "2026-06-30",
            "beta_asof_date": "2026-06-30",
            "universe": list(live_trade.SYMBOLS),
            "strategies": {sym: f"/tmp/{sym}.joblib" for sym in live_trade.SYMBOLS},
            "beta_by_symbol": {sym: 1.0 for sym in live_trade.SYMBOLS},
        }

        ok, reason = live_trade._check_manifest_freshness(
            manifest, pd.Timestamp("2026-07-01").date())

        self.assertFalse(ok)
        self.assertIn("not approved", reason)

        manifest["approved"] = True
        manifest["beta_by_symbol"].pop(live_trade.SYMBOLS[0])
        ok, reason = live_trade._check_manifest_freshness(
            manifest, pd.Timestamp("2026-07-01").date())

        self.assertFalse(ok)
        self.assertIn("beta missing", reason)


if __name__ == "__main__":
    unittest.main()
