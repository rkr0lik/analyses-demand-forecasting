"""Testy baseline'ów: poprawność na małym przykładzie oraz odporność na wyciek danych."""
import unittest

import pandas as pd

import config as cfg
from src.baselines import all_baselines, mean_window, naive, seasonal_naive
from src.data import daily_sales, load_raw, split_train_test
from src.features import daily_exog


def toy_series(n: int = 14) -> pd.Series:
    """Wartości 1..n w kolejnych dniach, żeby łatwo policzyć wynik w głowie."""
    return pd.Series(range(1, n + 1), index=pd.date_range("2024-01-01", periods=n), dtype=float)


class BaselineValuesTest(unittest.TestCase):
    def test_naive_repeats_last_day(self):
        f = naive(toy_series(), horizon=5)
        self.assertEqual(f.tolist(), [14.0] * 5)

    def test_seasonal_naive_repeats_last_week(self):
        f = seasonal_naive(toy_series(), horizon=14, season=7)
        self.assertEqual(f.tolist(), [8, 9, 10, 11, 12, 13, 14] * 2)

    def test_mean_uses_last_window_only(self):
        f = mean_window(toy_series(), horizon=3, window=4)
        self.assertEqual(f.tolist(), [12.5] * 3)  # (11 + 12 + 13 + 14) / 4

    def test_forecast_starts_the_day_after_training(self):
        f = naive(toy_series(), horizon=3)
        self.assertEqual(f.index[0], pd.Timestamp("2024-01-15"))
        self.assertEqual(len(f), 3)


@unittest.skipUnless(cfg.DATA_PATH.exists(), "brak pliku z danymi (demand_forecasting.csv)")
class BaselineLeakageTest(unittest.TestCase):
    def test_forecast_covers_test_period_exactly(self):
        raw = load_raw()
        split = split_train_test(daily_sales(raw), daily_exog(raw))
        f = all_baselines(split.y_train)
        self.assertTrue(f.index.equals(split.y_test.index))

    def test_changing_test_sales_does_not_change_forecasts(self):
        raw = load_raw()
        before = all_baselines(split_train_test(daily_sales(raw), daily_exog(raw)).y_train)

        altered = raw.copy()
        altered.loc[altered[cfg.COL_DATE] >= cfg.TEST_START, cfg.COL_TARGET] *= 1000
        after = all_baselines(split_train_test(daily_sales(altered), daily_exog(altered)).y_train)

        pd.testing.assert_frame_equal(before, after)


if __name__ == "__main__":
    unittest.main()
