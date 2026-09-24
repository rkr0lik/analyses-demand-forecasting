"""Testy ARIMAX na danych syntetycznych ze znanym efektem zmiennej zewnętrznej."""
import unittest

import numpy as np
import pandas as pd

import config as cfg
from src.arimax import cv_score, fit_forecast

EFFECT = 30.0


def synthetic(n: int, seed: int = cfg.RANDOM_STATE):
    """y = 100 + 30 * x + szum AR(1); x to flaga 0/1. Oś dat kończy się w TRAIN_END."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(end=cfg.TRAIN_END, periods=n, freq="D")
    x = rng.integers(0, 2, n).astype(float)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.5 * noise[t - 1] + rng.normal(0, 2)
    y = pd.Series(100 + EFFECT * x + noise, index=index)
    return y, pd.DataFrame({"epidemic": x}, index=index)


class FitForecastTest(unittest.TestCase):
    def setUp(self):
        y, X = synthetic(300)
        self.y, self.X = y[:-28], X[:-28]
        future = pd.date_range(self.y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
        self.future = lambda value: pd.DataFrame({"epidemic": value}, index=future)

    def test_forecast_has_future_index_and_no_nans(self):
        f = fit_forecast(self.y, self.X, self.future(0.0), order=(1, 0, 0))
        self.assertEqual(len(f), 28)
        self.assertTrue(f.index.equals(self.future(0.0).index))
        self.assertFalse(f.isna().any())

    def test_recovers_effect_of_external_variable(self):
        off = fit_forecast(self.y, self.X, self.future(0.0), order=(1, 0, 0))
        on = fit_forecast(self.y, self.X, self.future(1.0), order=(1, 0, 0))
        self.assertAlmostEqual((on - off).mean(), EFFECT, delta=3.0)

    def test_differenced_order_also_works(self):
        f = fit_forecast(self.y, self.X, self.future(1.0), order=(0, 1, 1))
        self.assertFalse(f.isna().any())


class CvScoreTest(unittest.TestCase):
    def test_returns_all_metrics_with_small_error(self):
        y, X = synthetic(cfg.CV_MIN_TRAIN_DAYS + cfg.CV_FOLDS * cfg.HORIZON)
        score = cv_score(y, X, order=(1, 0, 0))
        self.assertEqual(set(score), {"MAE", "RMSE", "MAPE", "Bias", "ostrzeżenia"})
        self.assertLess(score["MAE"], 5.0)  # szum ma odchylenie ok. 2, więc błąd musi być mały


if __name__ == "__main__":
    unittest.main()
