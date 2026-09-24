"""Testy bonusu: wybór najlepszych produktów tylko z treningu i spójność prognoz produktowych z całością."""
import unittest

import pandas as pd

import config as cfg
from src.bonus import (
    load_series_forecasts,
    ranking_comparison,
    product_daily_sales,
    product_forecasts,
    top_products,
    top_products_comparison,
    top_products_forecasts,
)
from src.data import daily_sales, load_raw
from src.features import daily_exog
from src.metrics import compute_metrics

HAVE_DATA = cfg.DATA_PATH.exists()
HAVE_FORECASTS = (cfg.OUTPUT_DIR / "forecasts_lgbm_series.csv").exists() and (cfg.OUTPUT_DIR / "forecasts_lgbm.csv").exists()


@unittest.skipUnless(HAVE_DATA, "brak pliku z danymi (demand_forecasting.csv)")
class TopProductsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()

    def test_top_three_match_decision_in_results_md(self):
        self.assertEqual(top_products(self.raw), ["P0007", "P0013", "P0004"])

    def test_selection_ignores_exam_sales(self):
        altered = self.raw.copy()
        in_exam = altered[cfg.COL_DATE] >= cfg.TEST_START
        altered.loc[in_exam & (altered[cfg.COL_PRODUCT] == "P0001"), cfg.COL_TARGET] *= 1000  # na egzaminie P0001 nagle sprzedaje najwięcej
        self.assertEqual(top_products(altered), top_products(self.raw))

    def test_product_sales_add_up_to_daily_total(self):
        by_product = product_daily_sales(self.raw)
        self.assertEqual(by_product.shape, (760, 20))
        self.assertTrue((by_product.sum(axis=1) == daily_sales(self.raw)).all())


@unittest.skipUnless(HAVE_FORECASTS, "brak prognoz LightGBM w outputs/ (uruchom: python -m src.run_models)")
class ProductForecastsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecasts = product_forecasts(load_series_forecasts())

    def test_shape_and_period(self):
        self.assertEqual(self.forecasts.shape, (cfg.HORIZON, 20))
        self.assertEqual(self.forecasts.index.min(), cfg.TEST_START)
        self.assertEqual(self.forecasts.index.max(), cfg.TEST_END)

    def test_products_add_up_to_lightgbm_daily_forecast(self):
        total = pd.read_csv(cfg.OUTPUT_DIR / "forecasts_lgbm.csv", index_col="date", parse_dates=True)["LightGBM"]
        pd.testing.assert_series_equal(self.forecasts.sum(axis=1), total, check_names=False, check_freq=False)


@unittest.skipUnless(HAVE_DATA and HAVE_FORECASTS, "brak danych albo prognoz LightGBM w outputs/")
class ProductComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.top = top_products(cls.raw)
        cls.forecasts = product_forecasts(load_series_forecasts())
        cls.epidemic = daily_exog(cls.raw).loc[cfg.TEST_START : cfg.TEST_END, "epidemic"]
        cls.table = top_products_comparison(cls.raw, cls.top, cls.forecasts, cls.epidemic)

    def test_has_four_models_for_each_of_three_products(self):
        self.assertEqual(len(self.table), 12)
        self.assertEqual(self.table.index.get_level_values("produkt").unique().tolist(), self.top)
        self.assertEqual(self.table.loc["P0007"].index.tolist(), ["naive", "seasonal naive", "średnia 28 dni", "LightGBM"])

    def test_lightgbm_metrics_match_direct_computation(self):
        sales = product_daily_sales(self.raw)
        for product in self.top:
            y_test = sales.loc[cfg.TEST_START : cfg.TEST_END, product]
            direct = compute_metrics(y_test, self.forecasts[product])
            for metric, value in direct.items():
                self.assertAlmostEqual(self.table.loc[(product, "LightGBM"), metric], value, places=6, msg=f"{product} {metric}")

    def test_naive_is_last_training_day_of_that_product(self):
        sales = product_daily_sales(self.raw)
        expected = sales.loc[cfg.TEST_START : cfg.TEST_END, "P0007"] - sales.loc[cfg.TRAIN_END, "P0007"]
        self.assertAlmostEqual(self.table.loc[("P0007", "naive"), "Bias"], -expected.mean(), places=6)

    def test_long_forecast_table_has_expected_layout_and_values(self):
        long = top_products_forecasts(self.raw, self.top, self.forecasts)
        self.assertEqual(len(long), len(self.top) * cfg.HORIZON)
        self.assertEqual(list(long.columns), ["date", "product", "actual", "naive", "seasonal naive", "średnia 28 dni", "LightGBM"])
        self.assertFalse(long.isna().any().any())
        sales = product_daily_sales(self.raw)
        for product, part in long.groupby("product"):
            self.assertEqual(part["actual"].tolist(), sales.loc[cfg.TEST_START : cfg.TEST_END, product].tolist())
            self.assertEqual(part["LightGBM"].tolist(), self.forecasts.loc[part["date"], product].tolist())

    def test_baselines_do_not_depend_on_exam_sales(self):
        altered = self.raw.copy()
        altered.loc[altered[cfg.COL_DATE] >= cfg.TEST_START, cfg.COL_TARGET] *= 1000
        table2 = top_products_comparison(altered, self.top, self.forecasts, self.epidemic)
        for product in self.top:
            for model in ("naive", "seasonal naive", "średnia 28 dni"):
                # prognoza się nie zmienia, więc bias zmienia się tylko o zmianę średniej rzeczywistej sprzedaży
                shift = (product_daily_sales(altered).loc[cfg.TEST_START : cfg.TEST_END, product].mean()
                         - product_daily_sales(self.raw).loc[cfg.TEST_START : cfg.TEST_END, product].mean())
                self.assertAlmostEqual(
                    table2.loc[(product, model), "Bias"], self.table.loc[(product, model), "Bias"] - shift, places=4
                )


@unittest.skipUnless(HAVE_DATA, "brak pliku z danymi (demand_forecasting.csv)")
class RankingComparisonTest(unittest.TestCase):
    """Wybór trójki zależy od okresu, z którego liczymy, i README ma to pokazywać."""

    @classmethod
    def setUpClass(cls):
        cls.ranking = ranking_comparison(load_raw())

    def test_both_rankings_are_present_and_consistent(self):
        for column in ("średnia z treningu", "średnia z całości", "pozycja (trening)", "pozycja (całość)"):
            self.assertIn(column, self.ranking.columns)
        best = self.ranking.sort_values("pozycja (trening)").index[0]
        self.assertEqual(best, top_products(load_raw())[0], "pierwszy produkt rankingu musi zgadzać się z wyborem")

    def test_choice_differs_between_periods(self):
        """README ma mówić, że z treningu i z całości danych wychodzą różne trójki."""
        from_train = self.ranking.nsmallest(3, "pozycja (trening)").index.tolist()
        from_whole = self.ranking.nsmallest(3, "pozycja (całość)").index.tolist()
        self.assertNotEqual(set(from_train), set(from_whole))

    def test_the_contested_places_are_almost_a_tie(self):
        """Na spornych miejscach różnice to pojedyncze sztuki dziennie."""
        contested = self.ranking[self.ranking["pozycja (trening)"].between(2, 4)]["średnia z treningu"]
        self.assertLess(contested.max() - contested.min(), 5.0)


if __name__ == "__main__":
    unittest.main()
