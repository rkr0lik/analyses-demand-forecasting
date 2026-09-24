"""Walidacja krocząca (walk-forward): "próbne egzaminy" na końcu okresu treningowego.

Okna testowe to kolejne bloki po `horizon` dni. Ostatni kończy się w TRAIN_END. W każdym foldzie model
uczy się na wszystkich danych sprzed okna testowego, więc okno uczenia rośnie z foldu na fold.
Dni po TRAIN_END (zbiór egzaminacyjny) nie są tu używane.
"""
from typing import NamedTuple

import pandas as pd

import config as cfg


class Fold(NamedTuple):
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def walk_forward_folds(
    index: pd.DatetimeIndex, horizon: int = cfg.HORIZON, n_folds: int = cfg.CV_FOLDS
) -> list[Fold]:
    """Podziały walk-forward na datach treningu, od najstarszego foldu do najnowszego."""
    assert index.max() <= cfg.TRAIN_END, "walidacja może używać tylko dat do końca treningu"
    first_test_start = len(index) - n_folds * horizon
    assert first_test_start >= cfg.CV_MIN_TRAIN_DAYS, "za mało danych na tyle foldów"

    folds = []
    for k in range(n_folds):
        start = first_test_start + k * horizon
        folds.append(Fold(train=index[:start], test=index[start : start + horizon]))
    return folds


if __name__ == "__main__":
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog

    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    for i, fold in enumerate(walk_forward_folds(split.y_train.index), start=1):
        epi_days = int(split.X_train.loc[fold.test, "epidemic"].sum())
        print(
            f"fold {i}: uczenie {len(fold.train)} dni (do {fold.train[-1].date()}), "
            f"test {fold.test[0].date()} – {fold.test[-1].date()}, dni epidemii w teście: {epi_days}"
        )
