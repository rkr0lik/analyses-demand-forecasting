"""Testy aplikacji Streamlit (streamlit.testing.v1.AppTest): wymagania zadania i działanie fragmentatorów."""
import unittest
from datetime import date
from pathlib import Path

import config as cfg
from src.app_data import TOTAL

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
NEEDED = ("daily_sales.csv", "product_daily_sales.csv", "forecasts_arimax.csv", "forecasts_ridge.csv", "forecasts_lgbm.csv",
          "forecasts_ensemble.csv", "interval_quantiles.csv", "scenarios.csv", "scenarios_top_products.csv",
          "forecasts_top_products.csv", "comparison.csv")
HAVE_OUTPUTS = all((cfg.OUTPUT_DIR / f).exists() for f in NEEDED)

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None


def run_app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def kpi_html(at) -> str:
    return " ".join(m.value for m in at.markdown if "kpi-value" in m.value)


@unittest.skipUnless(HAVE_OUTPUTS and AppTest is not None, "brak plików w outputs/ albo streamlit")
class AppTest_(unittest.TestCase):
    def test_starts_without_errors(self):
        at = run_app()
        self.assertEqual([e.value for e in at.exception], [])

    def test_meets_task_requirements_chart_metrics_with_baseline_and_date_slicer(self):
        at = run_app()
        self.assertGreaterEqual(len(at.get("plotly_chart")), 1, "brak wykresu z prognozą")
        start, end = at.slider(key="range").value  # fragmentator z wyborem zakresu dat
        self.assertIsInstance(start, date)
        self.assertLessEqual(start, end)
        metrics = at.dataframe[0].value  # tabela metryk
        self.assertEqual(list(metrics.index), ["ARIMAX", "naive", "seasonal naive", "średnia 28 dni"])
        self.assertTrue({"MAE", "RMSE", "MAPE (%)", "Bias"} <= set(metrics.columns))

    def test_default_metrics_match_saved_results(self):
        metrics = run_app().dataframe[0].value
        self.assertAlmostEqual(metrics.loc["ARIMAX", "MAE"], 405.8, delta=0.1)
        self.assertAlmostEqual(metrics.loc["średnia 28 dni", "MAE"], 1053.5, delta=0.1)

    def test_switching_to_product_view_offers_only_lightgbm(self):
        at = run_app()
        at.selectbox(key="view").select("P0007").run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(at.selectbox(key="model_P0007").options, ["LightGBM"])
        self.assertEqual(list(at.dataframe[0].value.index)[0], "LightGBM")

    def test_date_range_outside_forecast_shows_hint_and_dashes(self):
        at = run_app()
        at.slider(key="range").set_value((date(2023, 1, 1), date(2023, 6, 1))).run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(len(at.info), 1)
        self.assertIn("—", kpi_html(at))

    def test_narrow_range_changes_metrics_to_selected_days(self):
        at = run_app()
        at.slider(key="range").set_value((date(2024, 1, 3), date(2024, 1, 8))).run()  # tylko dni epidemii
        self.assertAlmostEqual(at.dataframe[0].value.loc["ARIMAX", "MAE"], 449.6, delta=0.1)

    def test_epidemic_scenario_changes_forecast_error(self):
        at = run_app()
        real_mae = at.dataframe[0].value.loc["ARIMAX", "MAE"]
        at.slider(key="epidemic_days").set_value(0).run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertGreater(at.dataframe[0].value.loc["ARIMAX", "MAE"], real_mae)  # scenariusz bez epidemii rozmija się z rzeczywistością

    def test_header_names_the_oracle_variant(self):
        at = run_app()
        self.assertTrue(any("Wariant oracle" in m.value for m in at.markdown))

    def test_every_model_in_total_view_runs(self):
        at = run_app()
        for model in at.selectbox(key=f"model_{TOTAL}").options:
            at.selectbox(key=f"model_{TOTAL}").select(model).run()
            self.assertEqual([e.value for e in at.exception], [], model)
            self.assertEqual(at.dataframe[0].value.index[0], model)


if __name__ == "__main__":
    unittest.main()
