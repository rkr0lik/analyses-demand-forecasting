"""Testy eksportu agregatów: zgodność sum z danymi i brak surowych kolumn w plikach."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import config as cfg
from src.data import load_raw
from src.export_actuals import export_actuals


@unittest.skipUnless(cfg.DATA_PATH.exists(), "brak pliku z danymi (demand_forecasting.csv)")
class ExportActualsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.tmp = tempfile.TemporaryDirectory()
        export_actuals(cls.raw, Path(cls.tmp.name))
        cls.daily = pd.read_csv(Path(cls.tmp.name) / "daily_sales.csv", index_col="date", parse_dates=True)
        cls.products = pd.read_csv(Path(cls.tmp.name) / "product_daily_sales.csv", index_col="date", parse_dates=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_daily_file_has_one_row_per_day_with_sales_and_epidemic_flag(self):
        self.assertEqual(list(self.daily.columns), ["units_sold", "epidemic"])
        self.assertEqual(len(self.daily), 760)

    def test_epidemic_flag_matches_raw_data(self):
        flag_from_raw = self.raw.groupby(cfg.COL_DATE)[cfg.COL_EPIDEMIC].first()
        self.assertTrue((self.daily["epidemic"] == flag_from_raw).all())
        self.assertEqual(self.daily["epidemic"].isin([0, 1]).all(), True)

    def test_totals_match_raw_data(self):
        self.assertEqual(self.daily["units_sold"].sum(), self.raw[cfg.COL_TARGET].sum())
        self.assertEqual(self.products.to_numpy().sum(), self.raw[cfg.COL_TARGET].sum())
        self.assertTrue((self.products.sum(axis=1) == self.daily["units_sold"]).all())

    def test_only_aggregates_are_saved(self):
        self.assertEqual(self.products.shape, (760, 20))
        forbidden = set(cfg.FORBIDDEN_FEATURES) | {"Price", cfg.COL_STORE, cfg.COL_WEATHER}
        for frame in (self.daily, self.products):
            self.assertEqual(forbidden & set(frame.columns), set())


if __name__ == "__main__":
    unittest.main()
