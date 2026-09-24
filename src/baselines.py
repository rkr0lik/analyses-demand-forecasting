"""Proste prognozy odniesienia. Dostają wyłącznie dane treningowe, więc nie widzą egzaminu.

Prognoza obejmuje `horizon` dni tuż po ostatnim dniu treningu.
"""
import numpy as np
import pandas as pd

import config as cfg


def _forecast_index(y_train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(y_train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def naive(y_train: pd.Series, horizon: int = cfg.HORIZON) -> pd.Series:
    """Naive: ostatni dzień treningu powtórzony `horizon` razy."""
    return pd.Series(y_train.iloc[-1], index=_forecast_index(y_train, horizon), dtype=float)


def seasonal_naive(y_train: pd.Series, horizon: int = cfg.HORIZON, season: int = cfg.SEASON_LENGTH) -> pd.Series:
    """Seasonal naive: ostatnie `season` dni treningu powtarzane cyklicznie (dla 28 dni: 4 razy po 7)."""
    values = np.resize(y_train.iloc[-season:].to_numpy(dtype=float), horizon)
    return pd.Series(values, index=_forecast_index(y_train, horizon))


def mean_window(y_train: pd.Series, horizon: int = cfg.HORIZON, window: int = cfg.MEAN_WINDOW) -> pd.Series:
    """Średnia z ostatnich `window` dni treningu, powtórzona `horizon` razy."""
    return pd.Series(y_train.iloc[-window:].mean(), index=_forecast_index(y_train, horizon), dtype=float)


def all_baselines(y_train: pd.Series, horizon: int = cfg.HORIZON) -> pd.DataFrame:
    """Prognozy wszystkich baseline'ów; jedna kolumna na metodę."""
    return pd.DataFrame(
        {
            "naive": naive(y_train, horizon),
            "seasonal naive": seasonal_naive(y_train, horizon),
            f"średnia {cfg.MEAN_WINDOW} dni": mean_window(y_train, horizon),
        }
    )


if __name__ == "__main__":
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog
    from src.metrics import metrics_table

    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    forecasts = all_baselines(split.y_train)
    for name in forecasts:
        print(f"\n{name}")
        print(metrics_table(split.y_test, forecasts[name], split.X_test["epidemic"]).round(1).to_string())
