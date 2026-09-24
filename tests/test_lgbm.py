"""Testy panelu i modelu LightGBM: poprawność lagów oraz brak wycieku danych."""
import unittest

import pandas as pd

import config as cfg
from src.data import daily_sales, load_raw
from src.lgbm import FEATURES, LAGS, build_panel, fit_predict
from src.validation import walk_forward_folds

SMALL_PARAMS = {"num_leaves": 7, "n_estimators": 20, "min_child_samples": 20}


@unittest.skipUnless(cfg.DATA_PATH.exists(), "brak pliku z danymi (demand_forecasting.csv)")
class LgbmPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.panel = build_panel(cls.raw)

    def test_no_forbidden_columns_or_price_in_features(self):
        self.assertEqual(set(FEATURES) & set(cfg.FORBIDDEN_FEATURES), set())
        self.assertFalse(any("price" in f.lower() for f in FEATURES))

    def test_panel_has_one_row_per_day_and_series(self):
        self.assertEqual(len(self.panel), len(self.raw))
        self.assertFalse(self.panel.duplicated(["date", "store", "product"]).any())

    def test_lags_match_raw_sales(self):
        store, product, day = "S001", "P0001", pd.Timestamp("2023-06-15")
        series = self.raw[(self.raw[cfg.COL_STORE] == store) & (self.raw[cfg.COL_PRODUCT] == product)]
        y = series.set_index(cfg.COL_DATE)[cfg.COL_TARGET]
        row = self.panel[
            (self.panel["store"] == store) & (self.panel["product"] == product) & (self.panel["date"] == day)
        ].iloc[0]
        self.assertEqual(row["lag_28"], y[day - pd.Timedelta(days=28)])
        self.assertAlmostEqual(row["roll7_lag28"], y[day - pd.Timedelta(days=34) : day - pd.Timedelta(days=28)].mean())
        self.assertAlmostEqual(row["roll28_lag28"], y[day - pd.Timedelta(days=55) : day - pd.Timedelta(days=28)].mean())

    def test_exam_window_has_complete_lag_features(self):
        exam = self.panel[self.panel["date"] >= cfg.TEST_START]
        self.assertFalse(exam[LAGS].isna().any().any())

    def test_changing_exam_sales_changes_neither_features_nor_forecast(self):
        altered = self.raw.copy()
        altered.loc[altered[cfg.COL_DATE] >= cfg.TEST_START, cfg.COL_TARGET] *= 1000
        panel2 = build_panel(altered)
        pd.testing.assert_frame_equal(panel2[["date", *FEATURES]], self.panel[["date", *FEATURES]])

        exam_days = pd.date_range(cfg.TEST_START, cfg.TEST_END)
        before = fit_predict(self.panel, cfg.TRAIN_END, exam_days, SMALL_PARAMS)
        after = fit_predict(panel2, cfg.TRAIN_END, exam_days, SMALL_PARAMS)
        pd.testing.assert_frame_equal(before, after)

    def test_fold_forecast_ignores_sales_after_the_fold_start(self):
        """Prognoza foldu z train_end = T nie może zależeć od sprzedaży po T (lag >= horyzont)."""
        fold = walk_forward_folds(daily_sales(self.raw)[: cfg.TRAIN_END].index)[2]
        end = fold.train[-1]
        altered = self.raw.copy()
        altered.loc[altered[cfg.COL_DATE] > end, cfg.COL_TARGET] *= 1000

        before = fit_predict(self.panel, end, fold.test, SMALL_PARAMS)
        after = fit_predict(build_panel(altered), end, fold.test, SMALL_PARAMS)
        pd.testing.assert_frame_equal(before, after)


if __name__ == "__main__":
    unittest.main()
