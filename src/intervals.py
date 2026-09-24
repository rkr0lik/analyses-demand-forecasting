"""Pasmo niepewności prognozy: empiryczne kwantyle błędów z walidacji kroczącej (tylko trening).

Błąd = rzeczywistość − prognoza. Pasmo 80% to [prognoza + kwantyl 10%, prognoza + kwantyl 90%] błędów
z prognoz out-of-fold. Jedno pasmo o stałej szerokości na model (błędów jest 168, za mało na osobne pasma
dla każdego dnia horyzontu). Zbiór egzaminacyjny służy tylko do podania pokrycia, nie do strojenia.
"""
import numpy as np
import pandas as pd

import config as cfg
from src.ensemble import ENSEMBLE_SIMPLE, ENSEMBLE_WEIGHTED, with_ensembles  # noqa: F401 (re-eksport)

LEVEL = 0.80


def error_quantiles(oof: pd.DataFrame, models: list[str], level: float = LEVEL) -> pd.DataFrame:
    """Dolny i górny przesuw pasma (w szt.) dla każdego modelu: kwantyle błędu (rzeczywistość − prognoza)."""
    assert oof.index.max() <= cfg.TRAIN_END, "pasmo może korzystać tylko z błędów z okresu treningowego"
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    rows = {}
    for model in models:
        error = oof["y"] - oof[model]
        rows[model] = {"dolny": np.quantile(error, lo), "górny": np.quantile(error, hi)}
    return pd.DataFrame(rows).T.rename_axis("model")


def bands(forecasts: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    """Forma długa: date, model, forecast, lower, upper dla modeli, które mają kwantyle."""
    frames = []
    for model in quantiles.index:
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "forecast": forecasts[model],
                    "lower": forecasts[model] + quantiles.loc[model, "dolny"],
                    "upper": forecasts[model] + quantiles.loc[model, "górny"],
                }
            )
        )
    return pd.concat(frames).rename_axis("date").reset_index()


def coverage_table(y_true: pd.Series, band_table: pd.DataFrame) -> pd.DataFrame:
    """Jaka część dni mieści się w paśmie oraz średnia szerokość pasma (szt.) dla każdego modelu."""
    rows = {}
    for model, part in band_table.groupby("model", sort=False):
        actual = y_true.loc[part["date"]].to_numpy()
        inside = (actual >= part["lower"].to_numpy()) & (actual <= part["upper"].to_numpy())
        rows[model] = {"pokrycie": inside.mean(), "średnia szerokość": (part["upper"] - part["lower"]).mean()}
    return pd.DataFrame(rows).T.rename_axis("model")


if __name__ == "__main__":
    from src.compare import load_forecasts
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog

    path = cfg.OUTPUT_DIR / "oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"brak {path}; uruchom najpierw: python -m src.oof")
    oof = with_ensembles(pd.read_csv(path, index_col="date", parse_dates=True))

    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    quantiles = error_quantiles(oof, [*cfg.MODELS, ENSEMBLE_SIMPLE, ENSEMBLE_WEIGHTED])
    band_table = bands(load_forecasts(split), quantiles)
    coverage = coverage_table(split.y_test, band_table)

    print(f"pasmo {LEVEL:.0%}: kwantyle błędów z {len(oof)} dni walidacji kroczącej (przesuw względem prognozy, szt.)")
    print(quantiles.round(0).to_string())
    print(f"\npokrycie na egzaminie (28 dni, informacyjnie; deklarowane {LEVEL:.0%}):")
    print(coverage.round(2).to_string())

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    quantiles.to_csv(cfg.OUTPUT_DIR / "interval_quantiles.csv")
    band_table.to_csv(cfg.OUTPUT_DIR / "intervals.csv", index=False)
    coverage.to_csv(cfg.OUTPUT_DIR / "interval_coverage.csv")
