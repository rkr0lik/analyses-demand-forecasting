"""Ensemble: zwykła i ważona średnia prognoz ARIMAX, LightGBM i Ridge.

Wagi (nieujemne, suma 1) dobieramy prostą siatką na prognozach out-of-fold z walidacji kroczącej,
minimalizując MAE. Zbiór egzaminacyjny nie bierze w tym udziału.
"""
import itertools

import numpy as np
import pandas as pd

import config as cfg

STEP = 0.05
ENSEMBLE_SIMPLE = "Ensemble zwykła średnia"
ENSEMBLE_WEIGHTED = "Ensemble ważona średnia"
MODELS = cfg.MODELS


def weight_grid(n_models: int, step: float = STEP) -> list[tuple]:
    """Wszystkie wektory wag z krokiem `step`, nieujemne, o sumie 1."""
    k = round(1 / step)
    return [tuple(c / k for c in combo) for combo in itertools.product(range(k + 1), repeat=n_models) if sum(combo) == k]


def weight_table(oof: pd.DataFrame, models=MODELS, step: float = STEP) -> pd.DataFrame:
    """Średnie MAE każdego wektora wag na prognozach out-of-fold; posortowane rosnąco po MAE."""
    P, y = oof[models].to_numpy(), oof["y"].to_numpy()
    grid = weight_grid(len(models), step)
    mae = [np.abs(P @ np.array(w) - y).mean() for w in grid]
    table = pd.DataFrame(grid, columns=models).assign(MAE=mae)
    return table.sort_values("MAE", kind="stable").reset_index(drop=True)


def best_weights(oof: pd.DataFrame, models=MODELS, step: float = STEP) -> dict:
    """Wagi o najniższym MAE na prognozach out-of-fold."""
    best = weight_table(oof, models, step).iloc[0]
    return {m: float(best[m]) for m in models}


def with_ensembles(preds: pd.DataFrame) -> pd.DataFrame:
    """Dodaj do prognoz składników dwie kolumny: zwykłą i ważoną średnią (wagi z config.py)."""
    return preds.assign(
        **{
            ENSEMBLE_SIMPLE: preds[cfg.MODELS].mean(axis=1),
            ENSEMBLE_WEIGHTED: combine(preds, cfg.ENSEMBLE_WEIGHTS),
        }
    )


def combine(preds: pd.DataFrame, weights: dict) -> pd.Series:
    """Średnia ważona kolumn `preds` (kolumny muszą odpowiadać kluczom `weights`)."""
    assert abs(sum(weights.values()) - 1) < 1e-9, "wagi muszą sumować się do 1"
    return sum(preds[m] * w for m, w in weights.items())


def leave_one_fold_out(oof: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    """Uczciwsza ocena wag: dla każdego foldu wagi z pozostałych foldów, MAE na tym foldzie.

    Kolumny: ważona (wagi z pozostałych foldów), zwykła średnia i każdy model osobno; wiersz = fold.
    """
    rows = {}
    for fold, test in oof.groupby("fold"):
        weights = best_weights(oof[oof["fold"] != fold], models)
        candidates = {"ważona": combine(test, weights), "zwykła średnia": test[models].mean(axis=1)}
        candidates.update({m: test[m] for m in models})
        rows[fold] = {name: (pred - test["y"]).abs().mean() for name, pred in candidates.items()}
    return pd.DataFrame(rows).T.rename_axis("fold")


if __name__ == "__main__":
    import config as cfg
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog
    from src.metrics import compute_metrics
    from src.oof import oof_predictions

    raw = load_raw()
    oof = oof_predictions(raw, split_train_test(daily_sales(raw), daily_exog(raw)))

    print("najlepsze wagi (MAE na prognozach out-of-fold):")
    print(weight_table(oof).head(8).round(3).to_string())
    weights = best_weights(oof)
    print(f"\nwybrane wagi: {weights}")

    print("\nprognozy out-of-fold, wszystkie foldy:")
    rows = {m: compute_metrics(oof["y"], oof[m]) for m in MODELS}
    rows["zwykła średnia"] = compute_metrics(oof["y"], oof[MODELS].mean(axis=1))
    rows["ważona"] = compute_metrics(oof["y"], combine(oof, weights))
    print(pd.DataFrame(rows).T.round(1).to_string())

    lofo = leave_one_fold_out(oof)
    print("\nMAE bez podglądania własnego foldu (wagi liczone z pozostałych foldów):")
    print(lofo.round(1).to_string())
    print("średnia po foldach:")
    print(lofo.mean().round(1).to_string())
