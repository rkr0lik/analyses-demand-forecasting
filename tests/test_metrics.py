"""Testy metryk na małym przykładzie liczonym ręcznie."""
import unittest

import numpy as np
import pandas as pd

from src.metrics import compute_metrics, metrics_table


class MetricsTest(unittest.TestCase):
    # Rzeczywiste 100, 200; prognozy 110, 180 -> błędy +10 i -20
    y_true = [100, 200]
    y_pred = [110, 180]

    def test_values(self):
        m = compute_metrics(self.y_true, self.y_pred)
        self.assertAlmostEqual(m["MAE"], 15.0)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(250))  # sqrt((100 + 400) / 2)
        self.assertAlmostEqual(m["MAPE"], 10.0)  # średnia z 10% i 10%
        self.assertAlmostEqual(m["Bias"], -5.0)  # (+10 - 20) / 2, ujemny = zaniża

    def test_overestimating_gives_positive_bias(self):
        self.assertGreater(compute_metrics([100, 100], [120, 130])["Bias"], 0)

    def test_zero_actual_is_rejected(self):
        with self.assertRaises(AssertionError):
            compute_metrics([0, 10], [1, 10])

    def test_table_splits_by_epidemic(self):
        idx = pd.date_range("2024-01-01", periods=4)
        y_true = pd.Series([100, 100, 200, 200], index=idx)
        y_pred = pd.Series([110, 110, 200, 180], index=idx)
        epidemic = pd.Series([1, 1, 0, 0], index=idx)
        table = metrics_table(y_true, y_pred, epidemic)
        self.assertEqual(table.loc["epidemia", "dni"], 2)
        self.assertAlmostEqual(table.loc["epidemia", "Bias"], 10.0)
        self.assertAlmostEqual(table.loc["bez epidemii", "Bias"], -10.0)
        self.assertAlmostEqual(table.loc["wszystkie dni", "MAE"], 10.0)

    def test_table_rejects_misaligned_index(self):
        a = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
        b = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-02", periods=2))
        with self.assertRaises(AssertionError):
            metrics_table(a, b, pd.Series([0, 0], index=a.index))


if __name__ == "__main__":
    unittest.main()
