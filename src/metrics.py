"""Jedna wspólna funkcja metryk dla wszystkich modeli i baseline'ów, żeby porównania były spójne.

Znak błędu: prognoza − rzeczywistość. Bias dodatni = model zawyża, ujemny = zaniża.
"""
import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, MAPE (w %) i bias dla dwóch ciągów o tej samej długości."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape and y_true.size > 0, "ciągi muszą mieć tę samą, niezerową długość"
    assert (y_true != 0).all(), "MAPE nie jest określone dla rzeczywistej sprzedaży równej 0"
    error = y_pred - y_true
    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt((error**2).mean()),
        "MAPE": (np.abs(error) / y_true).mean() * 100,
        "Bias": error.mean(),
    }


def metrics_table(y_true: pd.Series, y_pred: pd.Series, epidemic: pd.Series) -> pd.DataFrame:
    """Metryki dla wszystkich dni oraz osobno dla dni z epidemią i bez epidemii.

    Wszystkie trzy serie muszą mieć identyczny indeks (daty).
    """
    assert y_true.index.equals(y_pred.index) and y_true.index.equals(epidemic.index), "różne indeksy dat"
    groups = {
        "wszystkie dni": pd.Series(True, index=y_true.index),
        "epidemia": epidemic == 1,
        "bez epidemii": epidemic == 0,
    }
    rows = {}
    for name, mask in groups.items():
        if mask.any():
            rows[name] = {**compute_metrics(y_true[mask], y_pred[mask]), "dni": int(mask.sum())}
    return pd.DataFrame(rows).T
