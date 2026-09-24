"""Tabela porównawcza wszystkich modeli na zbiorze egzaminacyjnym (wariant oracle).

Prognozy modeli czyta z outputs/ (tworzą je skrypty run_*), baseline'y liczy od nowa z treningu.
Wszystkie metryki z jednej wspólnej funkcji (src/metrics.py).
"""
import pandas as pd

import config as cfg
from src.baselines import all_baselines
from src.data import Split, daily_sales, load_raw, split_train_test
from src.features import daily_exog
from src.metrics import metrics_table

# plik z prognozami -> (kolumny do użycia, skrypt, który go tworzy)
SOURCES = {
    "forecasts_arimax.csv": (["ARIMAX", "SARIMAX"], "src.run_models"),
    "forecasts_ridge.csv": (["Ridge"], "src.run_models"),
    "forecasts_lgbm.csv": (["LightGBM"], "src.run_models"),
    "forecasts_ensemble.csv": (["Ensemble zwykła średnia", "Ensemble ważona średnia"], "src.run_models"),
}
GROUPS = {
    "naive": "baseline", "seasonal naive": "baseline", f"średnia {cfg.MEAN_WINDOW} dni": "baseline",
    "ARIMAX": "eksperyment 1", "SARIMAX": "eksperyment 1 (porównawczy)",
    "Ridge": "eksperyment 2", "LightGBM": "eksperyment 2",
    "Ensemble zwykła średnia": "eksperyment 2", "Ensemble ważona średnia": "eksperyment 2",
}


def load_forecasts(split: Split) -> pd.DataFrame:
    """Prognozy egzaminacyjne wszystkich modeli: baseline'y policzone teraz, reszta z outputs/."""
    return forecasts_from_outputs(split.y_train, split.y_test.index)


def forecasts_from_outputs(y_train: pd.Series, exam_index: pd.DatetimeIndex) -> pd.DataFrame:
    """To samo co `load_forecasts`, ale wystarczy sprzedaż treningowa i oś dat egzaminu (bez surowych danych)."""
    frames = [all_baselines(y_train)]
    for filename, (columns, script) in SOURCES.items():
        path = cfg.OUTPUT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"brak {path}; uruchom najpierw: python -m {script}")
        frames.append(pd.read_csv(path, index_col="date", parse_dates=True)[columns])
    forecasts = pd.concat(frames, axis=1).reindex(exam_index)
    assert not forecasts.isna().any().any(), "prognozy nie pokrywają całego okresu egzaminacyjnego"
    assert list(forecasts.columns) == list(GROUPS), "nieoczekiwany zestaw modeli"
    return forecasts


def comparison_table(y_test: pd.Series, forecasts: pd.DataFrame, epidemic: pd.Series) -> pd.DataFrame:
    """Jeden wiersz na model: metryki dla wszystkich dni oraz osobno dla epidemii i bez epidemii."""
    rows = {}
    for model in forecasts:
        by_period = metrics_table(y_test, forecasts[model], epidemic)
        row = {"grupa": GROUPS[model], **by_period.loc["wszystkie dni", ["MAE", "RMSE", "MAPE", "Bias"]]}
        for period in ("epidemia", "bez epidemii"):
            for metric in ("MAE", "MAPE", "Bias"):
                row[f"{metric} {period}"] = by_period.loc[period, metric]
        rows[model] = row
    return pd.DataFrame(rows).T.rename_axis("model")


if __name__ == "__main__":
    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    table = comparison_table(split.y_test, load_forecasts(split), split.X_test["epidemic"])

    table.to_csv(cfg.OUTPUT_DIR / "comparison.csv")
    shown = table.sort_values("MAE")
    pd.set_option("display.width", 200)
    print(shown.astype({c: float for c in shown.columns if c != "grupa"}).round(1).to_string())
    print(f"\nzapisano: {cfg.OUTPUT_DIR / 'comparison.csv'}")
