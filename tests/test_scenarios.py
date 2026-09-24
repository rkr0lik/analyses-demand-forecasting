"""Testy scenariuszy epidemii: flaga w scenariuszu, zgodność N=6 z egzaminem i kształt plików."""
import unittest

import pandas as pd

import config as cfg
from src.data import daily_sales, load_raw, split_train_test
from src.features import daily_exog
from src.lgbm import build_panel
from src.oof import MODELS
from src.ensemble import ENSEMBLE_SIMPLE, ENSEMBLE_WEIGHTED, with_ensembles
from src.scenarios import epidemic_flags, scenario_components

REAL_EPIDEMIC_DAYS = 6  # 3-8.01.2024
OUT = cfg.OUTPUT_DIR
HAVE_DATA = cfg.DATA_PATH.exists()
HAVE_EXAM_FORECASTS = all((OUT / f).exists() for f in ("forecasts_arimax.csv", "forecasts_ridge.csv", "forecasts_lgbm.csv"))
HAVE_SCENARIOS = all((OUT / f).exists() for f in ("scenarios.csv", "scenarios_top_products.csv", "forecasts_top_products.csv"))


class EpidemicFlagsTest(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range(cfg.TEST_START, cfg.TEST_END)

    def test_first_n_days_are_epidemic(self):
        flags = epidemic_flags(self.index, 6)
        self.assertEqual(flags.tolist(), [1] * 6 + [0] * 22)
        self.assertTrue(flags.index.equals(self.index))

    def test_extremes(self):
        self.assertEqual(epidemic_flags(self.index, 0).sum(), 0)
        self.assertEqual(epidemic_flags(self.index, 28).sum(), 28)

    def test_out_of_range_is_rejected(self):
        for bad in (-1, 29):
            with self.assertRaises(AssertionError):
                epidemic_flags(self.index, bad)


class WithEnsemblesTest(unittest.TestCase):
    def test_adds_both_ensemble_columns(self):
        daily = pd.DataFrame({"ARIMAX": [100.0], "LightGBM": [10.0], "Ridge": [1000.0]})
        out = with_ensembles(daily)
        self.assertAlmostEqual(out[ENSEMBLE_SIMPLE].iloc[0], 370.0)
        w = cfg.ENSEMBLE_WEIGHTS
        self.assertAlmostEqual(out[ENSEMBLE_WEIGHTED].iloc[0], w["ARIMAX"] * 100 + w["LightGBM"] * 10 + w["Ridge"] * 1000)


@unittest.skipUnless(HAVE_DATA and HAVE_EXAM_FORECASTS, "brak danych albo prognoz egzaminacyjnych w outputs/")
class RealScenarioReproducesExamTest(unittest.TestCase):
    def test_scenario_with_real_epidemic_length_equals_exam_forecast(self):
        raw = load_raw()
        split = split_train_test(daily_sales(raw), daily_exog(raw))
        self.assertEqual(int(split.X_test["epidemic"].sum()), REAL_EPIDEMIC_DAYS)

        daily, _ = scenario_components(split, build_panel(raw), REAL_EPIDEMIC_DAYS)
        saved = {
            "ARIMAX": pd.read_csv(OUT / "forecasts_arimax.csv", index_col="date", parse_dates=True)["ARIMAX"],
            "Ridge": pd.read_csv(OUT / "forecasts_ridge.csv", index_col="date", parse_dates=True)["Ridge"],
            "LightGBM": pd.read_csv(OUT / "forecasts_lgbm.csv", index_col="date", parse_dates=True)["LightGBM"],
        }
        for model in MODELS:
            pd.testing.assert_series_equal(daily[model], saved[model], check_names=False, check_freq=False, rtol=1e-6)


@unittest.skipUnless(HAVE_SCENARIOS, "brak plików scenariuszy w outputs/ (uruchom: python -m src.scenarios)")
class SavedScenariosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = pd.read_csv(OUT / "scenarios.csv", parse_dates=["date"])
        cls.products = pd.read_csv(OUT / "scenarios_top_products.csv", parse_dates=["date"])

    def test_layout_of_daily_scenarios(self):
        self.assertEqual(len(self.daily), 29 * cfg.HORIZON * 5)
        self.assertEqual(sorted(self.daily["epidemic_days"].unique()), list(range(29)))
        self.assertEqual(self.daily["model"].nunique(), 5)
        self.assertFalse(self.daily["forecast"].isna().any())

    def test_more_epidemic_days_means_lower_total_forecast(self):
        totals = self.daily.groupby(["model", "epidemic_days"])["forecast"].sum().unstack("epidemic_days")
        for model, row in totals.iterrows():
            self.assertTrue(row.is_monotonic_decreasing, model)
            self.assertLess(row[28], 0.75 * row[0], model)  # pełna epidemia obniża sprzedaż o dziesiątki procent

    def test_real_scenario_of_top_products_equals_exam_forecast(self):
        exam = pd.read_csv(OUT / "forecasts_top_products.csv", parse_dates=["date"])
        real = self.products[self.products["epidemic_days"] == REAL_EPIDEMIC_DAYS]
        merged = exam.merge(real, on=["date", "product"], suffixes=("_exam", "_scenario"))
        self.assertEqual(len(merged), 3 * cfg.HORIZON)
        pd.testing.assert_series_equal(merged["LightGBM_exam"], merged["LightGBM_scenario"], check_names=False, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
