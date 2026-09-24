"""Testy logiki widoków aplikacji: spójność z zapisanymi wynikami, zakresy dat, okresy epidemii."""
import unittest

import pandas as pd

import config as cfg
from src.app_data import (
    TOTAL,
    build_view,
    epidemic_runs,
    epidemic_split,
    load_outputs,
    metrics_for_range,
    models_for,
    real_epidemic_days,
    views,
)

NEEDED = ("daily_sales.csv", "product_daily_sales.csv", "forecasts_arimax.csv", "forecasts_ridge.csv", "forecasts_lgbm.csv",
          "forecasts_ensemble.csv", "interval_quantiles.csv", "scenarios.csv", "scenarios_top_products.csv",
          "forecasts_top_products.csv", "comparison.csv")
HAVE_OUTPUTS = all((cfg.OUTPUT_DIR / f).exists() for f in NEEDED)
EXAM_START, EXAM_END = cfg.TEST_START, cfg.TEST_END


class EpidemicRunsTest(unittest.TestCase):
    def test_finds_contiguous_runs(self):
        index = pd.date_range("2024-01-01", periods=10)
        flag = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], index=index)
        self.assertEqual(
            epidemic_runs(flag),
            [(index[1], index[2]), (index[5], index[5]), (index[7], index[9])],
        )

    def test_no_epidemic_gives_no_runs(self):
        self.assertEqual(epidemic_runs(pd.Series([0, 0, 0], index=pd.date_range("2024-01-01", periods=3))), [])


class RealEpidemicDaysTest(unittest.TestCase):
    """Liczba dni epidemii na starcie prognozy jest liczona z danych, a nie wpisana na sztywno."""

    def frame(self, flags):
        return pd.DataFrame({"epidemic": flags}, index=pd.date_range(cfg.TEST_START, periods=cfg.HORIZON))

    def test_counts_leading_epidemic_days(self):
        self.assertEqual(real_epidemic_days(self.frame([1] * 6 + [0] * 22)), 6)

    def test_zero_when_no_epidemic(self):
        self.assertEqual(real_epidemic_days(self.frame([0] * cfg.HORIZON)), 0)

    def test_whole_horizon_in_epidemic(self):
        self.assertEqual(real_epidemic_days(self.frame([1] * cfg.HORIZON)), cfg.HORIZON)

    def test_rejects_epidemic_that_is_not_contiguous_from_the_start(self):
        with self.assertRaises(AssertionError):
            real_epidemic_days(self.frame([1, 1, 0, 1] + [0] * 24))


@unittest.skipUnless(HAVE_OUTPUTS, "brak plików w outputs/ (uruchom skrypty run_*, scenarios, intervals, export_actuals)")
class AppDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = load_outputs()

    def test_views_and_models(self):
        self.assertEqual(views(self.outputs), [TOTAL, "P0007", "P0013", "P0004"])

    def test_real_epidemic_days_matches_the_data(self):
        days = real_epidemic_days(self.outputs.daily)
        exam = self.outputs.daily.loc[pd.date_range(EXAM_START, EXAM_END), "epidemic"]
        self.assertEqual(days, int(exam.sum()), "epidemia na egzaminie musi być ciągła od pierwszego dnia")
        self.assertEqual(models_for("P0007"), ["LightGBM"])
        self.assertEqual(len(models_for(TOTAL)), 5)

    def test_unknown_model_for_product_is_rejected(self):
        with self.assertRaises(AssertionError):
            build_view(self.outputs, "P0007", "ARIMAX", 6)

    def test_real_scenario_equals_saved_exam_forecast(self):
        for model in models_for(TOTAL)[:3]:
            data = build_view(self.outputs, TOTAL, model, 6)
            pd.testing.assert_series_equal(data.forecast, self.outputs.forecasts[model], check_names=False, check_freq=False, rtol=1e-6)

    def test_band_surrounds_forecast(self):
        data = build_view(self.outputs, TOTAL, "ARIMAX", 6)
        self.assertTrue((data.band["lower"] < data.forecast).all() and (data.forecast < data.band["upper"]).all())
        self.assertIsNone(build_view(self.outputs, "P0007", "LightGBM", 6).band)

    def test_full_exam_metrics_match_saved_comparison(self):
        for model in models_for(TOTAL):
            table = metrics_for_range(build_view(self.outputs, TOTAL, model, 6), EXAM_START, EXAM_END)
            for metric in ("MAE", "RMSE", "MAPE", "Bias"):
                self.assertAlmostEqual(table.loc[model, metric], self.outputs.comparison.loc[model, metric], places=6, msg=f"{model} {metric}")

    def test_baselines_are_in_the_table_and_main_one_has_zero_delta(self):
        table = metrics_for_range(build_view(self.outputs, TOTAL, "Ridge", 6), EXAM_START, EXAM_END)
        self.assertEqual(list(table.index), ["Ridge", "naive", "seasonal naive", "średnia 28 dni"])
        self.assertEqual(table.loc["średnia 28 dni", "MAE vs średnia 28 dni"], 0.0)
        self.assertLess(table.loc["Ridge", "MAE vs średnia 28 dni"], 0)  # model lepszy od baseline'u

    def test_range_without_exam_days_gives_no_metrics(self):
        data = build_view(self.outputs, TOTAL, "ARIMAX", 6)
        self.assertIsNone(metrics_for_range(data, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01")))
        self.assertIsNone(epidemic_split(data, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01")))

    def test_narrow_range_uses_only_days_inside_it(self):
        data = build_view(self.outputs, TOTAL, "ARIMAX", 6)
        table = metrics_for_range(data, pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-08"))  # tylko 6 dni epidemii
        self.assertEqual(table.loc["ARIMAX", "dni"], 6)
        full_split = epidemic_split(data, EXAM_START, EXAM_END)
        self.assertAlmostEqual(table.loc["ARIMAX", "MAE"], full_split.loc["epidemia", "MAE"], places=6)

    def test_more_epidemic_days_lower_forecast_in_scenario(self):
        low = build_view(self.outputs, TOTAL, "LightGBM", 28).forecast.sum()
        high = build_view(self.outputs, TOTAL, "LightGBM", 0).forecast.sum()
        self.assertLess(low, high)

    def test_epidemic_history_matches_flag_file(self):
        runs = epidemic_runs(self.outputs.daily["epidemic"])
        self.assertEqual(sum((end - start).days + 1 for start, end in runs), 152)
        self.assertEqual(runs[-1][1], pd.Timestamp("2024-01-08"))


if __name__ == "__main__":
    unittest.main()
