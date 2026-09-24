"""Scenariusze epidemii: "epidemia trwa jeszcze N dni od początku okresu prognozy" (N = 0 … 28).

W scenariuszu N flaga epidemii to 1 w pierwszych N dniach prognozy i 0 w pozostałych; pozostałe zmienne
zewnętrzne zostają prawdziwe (wariant oracle). Modele uczą się wyłącznie na treningu, więc scenariusz zmienia
tylko dane podawane na dni prognozy. Prognozy liczone lokalnie i zapisane do outputs/, aplikacja ich nie przelicza.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import config as cfg
from src import arimax, ridge
from src.bonus import product_forecasts, top_products
from src.data import Split
from src.ensemble import ENSEMBLE_SIMPLE, ENSEMBLE_WEIGHTED, with_ensembles
from src.lgbm import build_panel, daily_total, fit_predict

MODELS = cfg.MODELS


def epidemic_flags(index: pd.DatetimeIndex, n_days: int) -> pd.Series:
    """Flaga epidemii: 1 w pierwszych `n_days` dniach okresu prognozy, potem 0."""
    assert 0 <= n_days <= len(index), "liczba dni epidemii poza zakresem horyzontu"
    return pd.Series((np.arange(len(index)) < n_days).astype(int), index=index)


def scenario_components(split: Split, panel: pd.DataFrame, n_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prognozy trzech modeli dziennych (kolumny MODELS) i LightGBM per seria dla scenariusza N = n_days."""
    exam_days = split.y_test.index
    flags = epidemic_flags(exam_days, n_days)
    X_future = split.X_test.assign(epidemic=flags)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # znane, nieszkodliwe (README.md)
        arimax_pred = arimax.fit_forecast(split.y_train, split.X_train, X_future, cfg.ARIMAX_ORDER)
    ridge_pred = ridge.fit_forecast(split.y_train, split.X_train, X_future, cfg.RIDGE_ALPHA)

    scenario_panel = panel.copy()
    in_exam = scenario_panel["date"].isin(exam_days)
    scenario_panel.loc[in_exam, "epidemic"] = scenario_panel.loc[in_exam, "date"].map(flags).astype(panel["epidemic"].dtype)
    per_series = fit_predict(scenario_panel, cfg.TRAIN_END, exam_days, cfg.LGBM_PARAMS)

    daily = pd.DataFrame(
        {"ARIMAX": arimax_pred, "LightGBM": daily_total(per_series).reindex(exam_days), "Ridge": ridge_pred}
    )[MODELS]
    return daily, per_series


def all_scenarios(raw: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wszystkie scenariusze N = 0 … HORIZON. Zwraca: prognozy dzienne (5 modeli) i LightGBM per produkt (top 3)."""
    assert split.X_test["epidemic"].tolist() == [1] * 6 + [0] * (cfg.HORIZON - 6), "egzamin: epidemia w pierwszych 6 dniach"
    panel, top = build_panel(raw), top_products(raw)
    daily_frames, product_frames = [], []
    for n_days in range(cfg.HORIZON + 1):
        daily, per_series = scenario_components(split, panel, n_days)
        long = with_ensembles(daily).rename_axis("date").reset_index().melt(
            id_vars="date", var_name="model", value_name="forecast"
        )
        daily_frames.append(long.assign(epidemic_days=n_days))
        by_product = product_forecasts(per_series)[top].rename_axis("date").reset_index().melt(
            id_vars="date", var_name="product", value_name="LightGBM"
        )
        product_frames.append(by_product.assign(epidemic_days=n_days))
    columns = ["epidemic_days", "date"]
    return (
        pd.concat(daily_frames)[[*columns, "model", "forecast"]].reset_index(drop=True),
        pd.concat(product_frames)[[*columns, "product", "LightGBM"]].reset_index(drop=True),
    )


if __name__ == "__main__":
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog

    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    daily_table, product_table = all_scenarios(raw, split)

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    daily_table.to_csv(cfg.OUTPUT_DIR / "scenarios.csv", index=False)
    product_table.to_csv(cfg.OUTPUT_DIR / "scenarios_top_products.csv", index=False)

    total = daily_table.groupby(["model", "epidemic_days"])["forecast"].sum().unstack("model")
    print("suma prognozy z 28 dni (szt.) w zależności od liczby dni epidemii:")
    print(total.loc[[0, 6, 14, 28]].round(0).astype(int).to_string())
    print("\nzmiana sumy 28 dni na każdy dodatkowy dzień epidemii (średnio, szt.):")
    print(((total.loc[28] - total.loc[0]) / 28).round(0).astype(int).to_string())
    print(f"\nzapisano: scenarios.csv ({len(daily_table)} wierszy), scenarios_top_products.csv ({len(product_table)} wierszy)")
