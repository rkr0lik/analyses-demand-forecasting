"""ARIMAX: regresja na zmiennych zewnętrznych + "pamięć" błędów (ARIMA).

Dobór rzędu (p, d, q) wyłącznie walidacją kroczącą na danych treningowych.
Prognozy w walidacji i na egzaminie dostają prawdziwe zmienne zewnętrzne (wariant oracle).

Uruchomienie siatki rzędów:
  python -m src.arimax             18 rzędów bez sezonowości
  python -m src.arimax sezonowe    20 kombinacji z sezonowością tygodniową (wariant SARIMAX)
"""
import sys
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config as cfg
from src.metrics import compute_metrics
from src.validation import walk_forward_folds

MAXITER = 500  # domyślne 50 iteracji nie wystarcza optymalizatorowi (ostrzeżenie o niezbieżności)
NO_SEASON = (0, 0, 0, 0)
ORDER_GRID = [(p, d, q) for p in (0, 1, 2) for d in (0, 1) for q in (0, 1, 2)]

# Wariant sezonowy (SARIMAX): czołówka rzędów nie-sezonowych × rzędy sezonowe o okresie tygodnia.
SEASON = 7
SEASONAL_BASE_ORDERS = [(0, 1, 1), (1, 1, 1), (0, 1, 2), (1, 1, 2)]
SEASONAL_ORDERS = [(1, 0, 0, SEASON), (0, 0, 1, SEASON), (1, 0, 1, SEASON), (0, 1, 1, SEASON), (1, 1, 1, SEASON)]


def fit_forecast(y_train, X_train, X_future, order, seasonal_order=NO_SEASON) -> pd.Series:
    """Naucz model na (y_train, X_train) i prognozuj dni z X_future."""
    scale = y_train.mean()  # sprzedaż ~9000 -> ~1, bo optymalizator statsmodels lepiej zbiega na małych liczbach
    # Stała tylko bez różnicowania; przy różnicowaniu stała oznaczałaby dryf.
    trend = "c" if order[1] == 0 and seasonal_order[1] == 0 else None
    model = SARIMAX(y_train / scale, exog=X_train, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit(disp=False, maxiter=MAXITER)
    forecast = result.forecast(steps=len(X_future), exog=X_future)
    return pd.Series(forecast.to_numpy() * scale, index=X_future.index)


def cv_score(y_train, X_train, order, seasonal_order=NO_SEASON) -> dict:
    """Średnie metryki z foldów walk-forward oraz liczba ostrzeżeń o niezbieżności optymalizatora."""
    fold_metrics, n_warnings = [], 0
    for fold in walk_forward_folds(y_train.index):
        end = fold.train[-1]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred = fit_forecast(y_train[:end], X_train[:end], X_train.loc[fold.test], order, seasonal_order)
        n_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        fold_metrics.append(compute_metrics(y_train.loc[fold.test], pred))
    mean = pd.DataFrame(fold_metrics).mean().to_dict()
    return {**mean, "ostrzeżenia": n_warnings}


def grid_search(y_train, X_train, orders=ORDER_GRID, seasonal_orders=(NO_SEASON,)) -> pd.DataFrame:
    """Oceń każdą kombinację rzędów walidacją kroczącą; wynik posortowany rosnąco po średnim MAE."""
    rows = [
        {
            "p": order[0], "d": order[1], "q": order[2],
            "P": seasonal[0], "D": seasonal[1], "Q": seasonal[2], "s": seasonal[3],
            **cv_score(y_train, X_train, order, seasonal),
        }
        for order in orders
        for seasonal in seasonal_orders
    ]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


if __name__ == "__main__":
    from src.baselines import mean_window
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog

    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    y, X = split.y_train, split.X_train

    if "sezonowe" in sys.argv:
        print(grid_search(y, X, SEASONAL_BASE_ORDERS, SEASONAL_ORDERS).round(1).to_string())
        sys.exit()

    table = grid_search(y, X)
    print(table.round(1).to_string())

    # Punkt odniesienia w tej samej walidacji: średnia z 28 dni.
    base_mae = [
        compute_metrics(y.loc[f.test], mean_window(y[: f.train[-1]]).to_numpy())["MAE"]
        for f in walk_forward_folds(y.index)
    ]
    print(f"\nśrednia 28 dni w tej samej walidacji: MAE = {sum(base_mae) / len(base_mae):.1f}")
