"""Testy Ridge na danych syntetycznych ze znanym efektem zmiennej zewnętrznej oraz testy cyklu rocznego."""
import unittest

import numpy as np
import pandas as pd

import config as cfg
from src.features import year_waves
from src.ridge import cv_score, fit_forecast, with_year_cycle
from tests.test_arimax import EFFECT, synthetic


class RidgeTest(unittest.TestCase):
    def setUp(self):
        y, X = synthetic(300)
        self.y, self.X = y[:-28], X[:-28]
        future = pd.date_range(self.y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
        self.future = lambda value: pd.DataFrame({"epidemic": value}, index=future)

    def effect(self, alpha):
        off = fit_forecast(self.y, self.X, self.future(0.0), alpha)
        on = fit_forecast(self.y, self.X, self.future(1.0), alpha)
        return (on - off).mean()

    def test_forecast_has_future_index(self):
        f = fit_forecast(self.y, self.X, self.future(0.0), alpha=1.0)
        self.assertEqual(len(f), 28)
        self.assertTrue(f.index.equals(self.future(0.0).index))

    def test_small_alpha_recovers_effect(self):
        self.assertAlmostEqual(self.effect(0.001), EFFECT, delta=3.0)

    def test_large_alpha_shrinks_effect(self):
        self.assertLess(self.effect(1e5), self.effect(0.001))

    def test_cv_score_returns_all_metrics(self):
        from src.validation import walk_forward_folds  # noqa: F401  (import sprawdza, że walidacja jest dostępna)
        import config as cfg

        y, X = synthetic(cfg.CV_MIN_TRAIN_DAYS + cfg.CV_FOLDS * cfg.HORIZON)
        score = cv_score(y, X, alpha=1.0)
        self.assertEqual(set(score), {"MAE", "RMSE", "MAPE", "Bias"})
        self.assertLess(score["MAE"], 10.0)


class YearCycleTest(unittest.TestCase):
    """Cykl roczny w Ridge: zależy wyłącznie od daty, więc nie może wnieść wycieku danych."""

    dates = pd.date_range("2022-01-01", periods=800, freq="D")  # dwa pełne cykle roczne

    def test_waves_depend_only_on_the_date(self):
        """Ten sam dzień roku daje te same fale, niezależnie od tego, co jest w danych."""
        same_day_2022 = year_waves(pd.DatetimeIndex(["2022-03-15"]))
        same_day_2023 = year_waves(pd.DatetimeIndex(["2023-03-15"]))
        np.testing.assert_allclose(same_day_2022.to_numpy(), same_day_2023.to_numpy(), atol=0.02)

    def test_waves_have_two_columns_per_pair_and_stay_bounded(self):
        w = year_waves(self.dates, waves=3)
        self.assertEqual(list(w.columns), ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3"])
        self.assertTrue((w.abs() <= 1.0).all().all())

    def test_waves_repeat_after_a_year(self):
        w = year_waves(self.dates, waves=1)
        np.testing.assert_allclose(w.loc["2022-06-01"].to_numpy(), w.loc["2023-06-01"].to_numpy(), atol=0.02)

    def test_zero_waves_leaves_features_untouched(self):
        X = pd.DataFrame({"epidemic": 0.0}, index=self.dates)
        pd.testing.assert_frame_equal(with_year_cycle(X, 0), X)

    def test_config_waves_add_expected_number_of_columns(self):
        X = pd.DataFrame({"epidemic": 0.0}, index=self.dates)
        widened = with_year_cycle(X, cfg.RIDGE_YEAR_WAVES)
        self.assertEqual(widened.shape[1], X.shape[1] + 2 * cfg.RIDGE_YEAR_WAVES)
        self.assertTrue(widened.notna().all().all())

    def test_forecast_uses_the_cycle_by_default(self):
        """Prognoza z cyklem różni się od prognozy bez niego — czyli cykl faktycznie wchodzi do modelu."""
        y, X = synthetic(400)
        future = pd.DataFrame({"epidemic": 0.0}, index=pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=28))
        with_cycle = fit_forecast(y, X, future, alpha=1.0)
        without = fit_forecast(y, X, future, alpha=1.0, waves=0)
        self.assertGreater((with_cycle - without).abs().max(), 0.0)


if __name__ == "__main__":
    unittest.main()
