"""Ridge: regresja liniowa na dziennych zmiennych zewnętrznych z "hamulcem" na współczynniki.

Nie ma części autoregresyjnej, więc to najprostszy składnik ensemble. Oprócz zmiennych zewnętrznych
dostaje cykl roczny (fale sin/cos dnia roku), bo tej sezonowości zmienne zewnętrzne nie tłumaczą.
ARIMAX dostaje ten sam X_train, ale bez fal. Alpha i liczbę par fal wybrała walidacja krocząca na treningu.
"""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from src.features import year_waves
from src.metrics import compute_metrics
from src.validation import walk_forward_folds

ALPHA_GRID = [0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 1000]
WAVES_GRID = [0, 1, 2, 3, 4]  # 0 = bez cyklu rocznego


def with_year_cycle(X: pd.DataFrame, waves: int) -> pd.DataFrame:
    """Zmienne zewnętrzne i fale cyklu rocznego. Fale zależą tylko od daty, więc nie ma tu wycieku."""
    return X if waves == 0 else pd.concat([X, year_waves(X.index, waves)], axis=1)


def fit_forecast(y_train, X_train, X_future, alpha, waves: int = None) -> pd.Series:
    """Naucz Ridge na (y_train, X_train) i prognozuj dni z X_future. Cechy są standaryzowane."""
    waves = cfg.RIDGE_YEAR_WAVES if waves is None else waves
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=cfg.RANDOM_STATE))
    model.fit(with_year_cycle(X_train, waves), y_train)
    future = with_year_cycle(X_future, waves)
    return pd.Series(model.predict(future), index=future.index)


def cv_score(y_train, X_train, alpha, waves: int = None) -> dict:
    """Średnie metryki z foldów walk-forward."""
    fold_metrics = []
    for fold in walk_forward_folds(y_train.index):
        end = fold.train[-1]
        pred = fit_forecast(y_train[:end], X_train[:end], X_train.loc[fold.test], alpha, waves)
        fold_metrics.append(compute_metrics(y_train.loc[fold.test], pred))
    return pd.DataFrame(fold_metrics).mean().to_dict()


def grid_search(y_train, X_train, alphas=ALPHA_GRID, waves_grid=WAVES_GRID) -> pd.DataFrame:
    """Oceń każdą parę (alpha, liczba par fal) walidacją kroczącą; wynik posortowany rosnąco po średnim MAE."""
    rows = [{"alpha": alpha, "fale": waves, **cv_score(y_train, X_train, alpha, waves)}
            for waves in waves_grid for alpha in alphas]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


if __name__ == "__main__":
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog

    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    print(grid_search(split.y_train, split.X_train).round(2).to_string())
