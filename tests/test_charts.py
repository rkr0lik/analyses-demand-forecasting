"""Testy wykresu: składniki, jedna oś Y, legenda, zachowanie przy zakresie bez prognozy."""
import unittest

import pandas as pd

import config as cfg
from app.charts import ACTUAL, FORECAST, build_figure
from src.app_data import TOTAL, build_view, load_outputs

NEEDED = ("daily_sales.csv", "product_daily_sales.csv", "forecasts_arimax.csv", "forecasts_ridge.csv", "forecasts_lgbm.csv",
          "forecasts_ensemble.csv", "interval_quantiles.csv", "scenarios.csv", "scenarios_top_products.csv",
          "forecasts_top_products.csv", "comparison.csv")
HAVE_OUTPUTS = all((cfg.OUTPUT_DIR / f).exists() for f in NEEDED)


@unittest.skipUnless(HAVE_OUTPUTS, "brak plików w outputs/")
class FigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = load_outputs()
        cls.total = build_view(cls.outputs, TOTAL, "ARIMAX", 6)
        cls.product = build_view(cls.outputs, "P0007", "LightGBM", 6)

    def legend_names(self, fig):
        return [t.name for t in fig.data if t.showlegend is not False]

    def test_total_view_has_actual_forecast_band_baseline_and_epidemic_in_legend(self):
        fig = build_figure(self.total, pd.Timestamp("2023-10-05"), cfg.TEST_END)
        self.assertEqual(
            self.legend_names(fig),
            ["Sprzedaż rzeczywista", "Prognoza: ARIMAX", "Pasmo niepewności 80%", "Baseline: średnia 28 dni", "Dni z epidemią"],
        )

    def test_uses_the_validated_colors_for_actual_and_forecast(self):
        fig = build_figure(self.total, pd.Timestamp("2023-10-05"), cfg.TEST_END)
        colors = {t.name: t.line.color for t in fig.data if t.line and t.line.color}
        self.assertEqual(colors["Sprzedaż rzeczywista"], ACTUAL)
        self.assertEqual(colors["Prognoza: ARIMAX"], FORECAST)

    def test_single_y_axis(self):
        fig = build_figure(self.total, pd.Timestamp("2023-10-05"), cfg.TEST_END)
        self.assertNotIn("yaxis2", fig.layout)

    def test_product_view_has_no_band(self):
        fig = build_figure(self.product, pd.Timestamp("2023-10-05"), cfg.TEST_END)
        self.assertNotIn("Pasmo niepewności 80%", self.legend_names(fig))
        self.assertIn("Prognoza: LightGBM", self.legend_names(fig))

    def test_range_without_forecast_shows_only_history(self):
        fig = build_figure(self.total, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01"))
        self.assertNotIn("Prognoza: ARIMAX", self.legend_names(fig))
        self.assertEqual(self.legend_names(fig)[0], "Sprzedaż rzeczywista")

    def test_forecast_line_covers_only_exam_days(self):
        fig = build_figure(self.total, pd.Timestamp("2023-10-05"), cfg.TEST_END)
        forecast = next(t for t in fig.data if t.name == "Prognoza: ARIMAX")
        self.assertEqual(len(forecast.x), cfg.HORIZON)
        self.assertEqual(pd.Timestamp(forecast.x[0]), cfg.TEST_START)


if __name__ == "__main__":
    unittest.main()
