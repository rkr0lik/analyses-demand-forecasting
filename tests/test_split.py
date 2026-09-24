"""Testy podziału na trening i egzamin oraz ochrony przed wyciekiem danych.

Uruchomienie z katalogu głównego: python -m unittest discover -v
"""
import unittest

import pandas as pd

import config as cfg
from src.data import daily_sales, load_raw, split_train_test
from src.features import EXOG_COLUMNS, daily_exog


@unittest.skipUnless(cfg.DATA_PATH.exists(), "brak pliku z danymi (demand_forecasting.csv)")
class SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.split = split_train_test(daily_sales(cls.raw), daily_exog(cls.raw))

    def test_boundary_dates(self):
        s = self.split
        self.assertEqual(s.y_train.index.max(), cfg.TRAIN_END)
        self.assertEqual(s.y_test.index.min(), cfg.TEST_START)
        self.assertEqual(s.y_test.index.max(), cfg.TEST_END)

    def test_test_has_horizon_days(self):
        self.assertEqual(len(self.split.y_test), cfg.HORIZON)

    def test_no_shared_dates(self):
        self.assertTrue(self.split.y_train.index.intersection(self.split.y_test.index).empty)

    def test_features_aligned_with_target(self):
        s = self.split
        self.assertTrue(s.X_train.index.equals(s.y_train.index))
        self.assertTrue(s.X_test.index.equals(s.y_test.index))

    def test_no_forbidden_columns_in_features(self):
        self.assertEqual(set(EXOG_COLUMNS) & set(cfg.FORBIDDEN_FEATURES), set())
        self.assertFalse(any("price" in col.lower() for col in EXOG_COLUMNS), "Price jest poza modelami")

    def test_discount_is_not_a_feature_anywhere(self):
        """Rabat powiela promocję, więc nie może trafić ani do modeli dziennych, ani do LightGBM."""
        from src.lgbm import FEATURES

        self.assertFalse(any("discount" in col.lower() for col in EXOG_COLUMNS), EXOG_COLUMNS)
        self.assertFalse(any("discount" in col.lower() for col in FEATURES), FEATURES)
        self.assertNotIn("discount_mean", self.split.X_train.columns)

    def test_changing_test_sales_does_not_change_train_or_features(self):
        """Jeśli cechy albo trening zależą od sprzedaży z egzaminu, ta zmiana coś w nich przesunie."""
        altered = self.raw.copy()
        in_test = altered[cfg.COL_DATE] >= cfg.TEST_START
        altered.loc[in_test, cfg.COL_TARGET] *= 1000

        s = split_train_test(daily_sales(altered), daily_exog(altered))
        pd.testing.assert_series_equal(s.y_train, self.split.y_train)
        pd.testing.assert_frame_equal(daily_exog(altered), daily_exog(self.raw))
        self.assertGreater(s.y_test.sum(), 100 * self.split.y_test.sum(), "zmiana testowa nie zadziałała")


if __name__ == "__main__":
    unittest.main()
