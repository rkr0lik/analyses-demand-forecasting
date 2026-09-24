"""Końcowe prognozy wszystkich modeli na zbiór egzaminacyjny (wariant oracle).

Modele uczą się tylko na treningu, parametry i wagi biorą z config.py. Skrypt zapisuje do outputs/
pięć plików:
forecasts_arimax.csv, forecasts_ridge.csv, forecasts_lgbm.csv, forecasts_lgbm_series.csv i forecasts_ensemble.csv.

Użycie: python -m src.run_models
"""
import pandas as pd

import config as cfg
from src import arimax, ridge
from src.data import Split, daily_sales, load_raw, split_train_test
from src.ensemble import ENSEMBLE_SIMPLE, ENSEMBLE_WEIGHTED, combine
from src.features import daily_exog
from src.lgbm import build_panel, daily_total, fit_predict
from src.metrics import metrics_table


def arimax_forecasts(split: Split) -> pd.DataFrame:
    """Prognozy ARIMAX i jego wariantu sezonowego SARIMAX na dni egzaminacyjne."""
    return pd.DataFrame(
        {
            "ARIMAX": arimax.fit_forecast(split.y_train, split.X_train, split.X_test, cfg.ARIMAX_ORDER),
            "SARIMAX": arimax.fit_forecast(
                split.y_train, split.X_train, split.X_test, cfg.SARIMAX_ORDER, cfg.SARIMAX_SEASONAL_ORDER
            ),
        }
    )


def ridge_forecast(split: Split) -> pd.Series:
    """Prognoza Ridge na dni egzaminacyjne."""
    return ridge.fit_forecast(split.y_train, split.X_train, split.X_test, cfg.RIDGE_ALPHA).rename("Ridge")


def lgbm_forecasts(raw: pd.DataFrame, split: Split) -> tuple[pd.Series, pd.DataFrame]:
    """Prognozy globalnego LightGBM: suma dzienna i prognozy dla każdej serii sklep × produkt."""
    exam_days = split.y_test.index
    per_series = fit_predict(build_panel(raw), cfg.TRAIN_END, exam_days, cfg.LGBM_PARAMS)
    return daily_total(per_series).reindex(exam_days).rename("LightGBM"), per_series


def ensemble_forecasts(components: pd.DataFrame) -> pd.DataFrame:
    """Prognozy składników ensemble oraz ich zwykła i ważona średnia."""
    return components.assign(
        **{
            ENSEMBLE_SIMPLE: components.mean(axis=1),
            ENSEMBLE_WEIGHTED: combine(components, cfg.ENSEMBLE_WEIGHTS),
        }
    )


def all_forecasts(raw: pd.DataFrame, split: Split) -> dict[str, pd.DataFrame]:
    """Wszystkie prognozy egzaminacyjne w słowniku {nazwa pliku: tabela}."""
    arimax_part = arimax_forecasts(split)
    ridge_part = ridge_forecast(split)
    lgbm_daily, lgbm_series = lgbm_forecasts(raw, split)
    components = pd.DataFrame({"ARIMAX": arimax_part["ARIMAX"], "LightGBM": lgbm_daily, "Ridge": ridge_part})[cfg.MODELS]
    return {
        "forecasts_arimax.csv": arimax_part,
        "forecasts_ridge.csv": ridge_part.to_frame(),
        "forecasts_lgbm.csv": lgbm_daily.to_frame(),
        "forecasts_lgbm_series.csv": lgbm_series,
        "forecasts_ensemble.csv": ensemble_forecasts(components),
    }


if __name__ == "__main__":
    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    saved = all_forecasts(raw, split)

    epidemic = split.X_test["epidemic"]
    for name in ("forecasts_arimax.csv", "forecasts_ridge.csv", "forecasts_lgbm.csv", "forecasts_ensemble.csv"):
        for model in saved[name]:
            print(f"\n{model}")
            print(metrics_table(split.y_test, saved[name][model], epidemic).round(1).to_string())

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, frame in saved.items():
        if filename == "forecasts_lgbm_series.csv":
            frame.to_csv(cfg.OUTPUT_DIR / filename, index=False)
        else:
            frame.to_csv(cfg.OUTPUT_DIR / filename, index_label="date")
    print(f"\nzapisano {len(saved)} plików w {cfg.OUTPUT_DIR}")
