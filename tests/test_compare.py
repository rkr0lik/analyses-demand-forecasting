"""Testy tabeli porównawczej: liczba modeli, spójność z metrykami liczonymi wprost, brak wycieku w baseline'ach."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from src.compare import GROUPS, SOURCES, comparison_table, load_forecasts
from src.data import daily_sales, load_raw, split_train_test
from src.features import daily_exog
from src.metrics import compute_metrics

# MAE z README.md (egzamin, wariant oracle). Po zmianie modelu trzeba tu wpisać nowe liczby.
EXPECTED_MAE = {
    "naive": 2460.1, "seasonal naive": 2595.4, "średnia 28 dni": 1053.5,
    "ARIMAX": 405.8, "SARIMAX": 409.3,
    "Ridge": 374.0, "LightGBM": 390.7,
    "Ensemble zwykła średnia": 389.6, "Ensemble ważona średnia": 383.5,
}

HAVE_DATA = cfg.DATA_PATH.exists()
HAVE_FORECASTS = all((cfg.OUTPUT_DIR / name).exists() for name in SOURCES)


@unittest.skipUnless(HAVE_DATA and HAVE_FORECASTS, "brak danych albo prognoz w outputs/ (uruchom skrypty run_*)")
class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.split = split_train_test(daily_sales(cls.raw), daily_exog(cls.raw))
        cls.forecasts = load_forecasts(cls.split)
        cls.epidemic = cls.split.X_test["epidemic"]
        cls.table = comparison_table(cls.split.y_test, cls.forecasts, cls.epidemic)

    def test_has_nine_models(self):
        self.assertEqual(len(self.table), 9)
        self.assertEqual(list(self.table.index), list(GROUPS))

    def test_forecasts_cover_exam_period_without_gaps(self):
        self.assertTrue(self.forecasts.index.equals(self.split.y_test.index))
        self.assertFalse(self.forecasts.isna().any().any())

    def test_metrics_match_direct_computation(self):
        y = self.split.y_test
        for model in self.forecasts:
            direct = compute_metrics(y, self.forecasts[model])
            for metric, value in direct.items():
                self.assertAlmostEqual(self.table.loc[model, metric], value, places=6, msg=f"{model} {metric}")
            for period, mask in (("epidemia", self.epidemic == 1), ("bez epidemii", self.epidemic == 0)):
                part = compute_metrics(y[mask], self.forecasts.loc[mask, model])
                self.assertAlmostEqual(self.table.loc[model, f"MAE {period}"], part["MAE"], places=6)
                self.assertAlmostEqual(self.table.loc[model, f"Bias {period}"], part["Bias"], places=6)

    def test_mae_matches_results_md(self):
        for model, expected in EXPECTED_MAE.items():
            self.assertAlmostEqual(self.table.loc[model, "MAE"], expected, delta=0.1, msg=model)

    def test_baselines_do_not_depend_on_exam_sales(self):
        altered = self.raw.copy()
        altered.loc[altered[cfg.COL_DATE] >= cfg.TEST_START, cfg.COL_TARGET] *= 1000
        split2 = split_train_test(daily_sales(altered), daily_exog(altered))
        baselines = list(self.forecasts.columns[:3])
        self.assertTrue(self.forecasts[baselines].equals(load_forecasts(split2)[baselines]))


@unittest.skipUnless(HAVE_DATA, "brak pliku z danymi (demand_forecasting.csv)")
class MissingForecastsTest(unittest.TestCase):
    def test_missing_file_gives_hint_which_script_to_run(self):
        raw = load_raw()
        split = split_train_test(daily_sales(raw), daily_exog(raw))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cfg, "OUTPUT_DIR", Path(tmp)):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_forecasts(split)
        self.assertIn("python -m src.run_models", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
