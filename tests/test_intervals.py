"""Testy pasma niepewności na małych przykładach oraz zgodności zapisanych plików z danymi wejściowymi."""
import unittest

import numpy as np
import pandas as pd

import config as cfg
from src.intervals import bands, coverage_table, error_quantiles, with_ensembles
from src.oof import MODELS


def toy_oof(errors, end=cfg.TRAIN_END) -> pd.DataFrame:
    """Rzeczywistość 1000; model A myli się dokładnie o `errors` (błąd = rzeczywistość - prognoza).

    Oś dat kończy się w `end` (domyślnie w ostatnim dniu treningu).
    """
    index = pd.date_range(end=end, periods=len(errors))
    y = pd.Series(1000.0, index=index)
    return pd.DataFrame({"y": y, "A": y - np.asarray(errors, dtype=float)}, index=index)


class ErrorQuantilesTest(unittest.TestCase):
    def test_quantiles_of_symmetric_errors_are_symmetric(self):
        oof = toy_oof(np.arange(-50, 51))  # błędy od -50 do 50
        q = error_quantiles(oof, ["A"], level=0.80)
        self.assertAlmostEqual(q.loc["A", "dolny"], -40.0)
        self.assertAlmostEqual(q.loc["A", "górny"], 40.0)

    def test_higher_level_gives_wider_band(self):
        oof = toy_oof(np.arange(-50, 51))
        narrow, wide = error_quantiles(oof, ["A"], 0.5), error_quantiles(oof, ["A"], 0.9)
        self.assertGreater(wide.loc["A", "górny"], narrow.loc["A", "górny"])
        self.assertLess(wide.loc["A", "dolny"], narrow.loc["A", "dolny"])

    def test_rejects_errors_from_exam_period(self):
        with self.assertRaises(AssertionError):
            error_quantiles(toy_oof(np.arange(10), end=cfg.TEST_END), ["A"])


class BandsAndCoverageTest(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-03", periods=3)
        self.forecasts = pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=self.index)
        self.quantiles = pd.DataFrame({"dolny": [-10.0], "górny": [20.0]}, index=pd.Index(["A"], name="model"))

    def test_band_is_forecast_shifted_by_quantiles(self):
        b = bands(self.forecasts, self.quantiles)
        self.assertEqual(b["lower"].tolist(), [90.0, 190.0, 290.0])
        self.assertEqual(b["upper"].tolist(), [120.0, 220.0, 320.0])
        self.assertTrue((b["lower"] <= b["forecast"]).all() and (b["forecast"] <= b["upper"]).all())

    def test_coverage_counts_days_inside_band(self):
        y = pd.Series([105.0, 250.0, 310.0], index=self.index)  # dzień 2 poza pasmem
        table = coverage_table(y, bands(self.forecasts, self.quantiles))
        self.assertAlmostEqual(table.loc["A", "pokrycie"], 2 / 3)
        self.assertAlmostEqual(table.loc["A", "średnia szerokość"], 30.0)


class EnsembleColumnsTest(unittest.TestCase):
    def test_simple_and_weighted_ensemble_columns(self):
        index = pd.date_range("2023-12-01", periods=2)
        oof = pd.DataFrame({"y": [1.0, 1.0], "ARIMAX": [100.0, 200.0], "LightGBM": [10.0, 20.0], "Ridge": [1000.0, 2000.0]}, index=index)
        out = with_ensembles(oof)
        self.assertEqual(out["Ensemble zwykła średnia"].tolist(), oof[MODELS].mean(axis=1).tolist())
        w = cfg.ENSEMBLE_WEIGHTS
        expected = w["ARIMAX"] * 100 + w["LightGBM"] * 10 + w["Ridge"] * 1000
        self.assertAlmostEqual(out["Ensemble ważona średnia"].iloc[0], expected)


HAVE_OUTPUTS = all((cfg.OUTPUT_DIR / f).exists() for f in ("oof_predictions.csv", "interval_quantiles.csv", "intervals.csv"))


@unittest.skipUnless(HAVE_OUTPUTS, "brak plików w outputs/ (uruchom: python -m src.intervals)")
class SavedFilesTest(unittest.TestCase):
    def test_saved_quantiles_match_recomputation_from_oof(self):
        oof = with_ensembles(pd.read_csv(cfg.OUTPUT_DIR / "oof_predictions.csv", index_col="date", parse_dates=True))
        saved = pd.read_csv(cfg.OUTPUT_DIR / "interval_quantiles.csv", index_col="model")
        fresh = error_quantiles(oof, list(saved.index))
        pd.testing.assert_frame_equal(saved, fresh, check_names=False)

    def test_saved_bands_cover_five_models_and_exam_days(self):
        table = pd.read_csv(cfg.OUTPUT_DIR / "intervals.csv", parse_dates=["date"])
        self.assertEqual(len(table), 5 * cfg.HORIZON)
        self.assertEqual(table["date"].min(), cfg.TEST_START)
        self.assertEqual(table["date"].max(), cfg.TEST_END)
        self.assertTrue((table["lower"] < table["upper"]).all())


if __name__ == "__main__":
    unittest.main()
