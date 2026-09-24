"""Globalny LightGBM: jeden model dla wszystkich 100 serii (sklep × produkt), prognozy sumowane do dnia.

Ochrona przed wyciekiem danych:
- cechy ze sprzedaży to lag co najmniej HORIZON (28) dni oraz średnie kroczące liczone od tego lagu,
  więc dla każdego dnia prognozy sięgają tylko do dni sprzed początku 28-dniowego okna prognozy
  (dotyczy walidacji kroczącej i egzaminu), bez prognozy rekurencyjnej;
- sprzedaż z okresu po TRAIN_END jest przed budową cech zamieniana na NaN, więc nie ma jak wejść do cech;
- cena, rabat, Demand, Inventory Level, Units Ordered i ceny konkurencji nie są cechami.
"""
import itertools

import lightgbm as lgb
import pandas as pd

import config as cfg
from src.metrics import compute_metrics
from src.validation import walk_forward_folds

LAG = cfg.HORIZON
assert LAG >= cfg.HORIZON, "lag krótszy niż horyzont oznacza wyciek danych"

# Cechy kategoryczne: identyfikacja serii i pogoda sklepu (znana lub prognozowana z góry, wariant oracle).
CATEGORICAL = ["store", "product", "category", "region", "weather"]
# Zmienne zewnętrzne na poziomie wiersza: epidemia i to, czy produkt jest w promocji.
# Rabatu nie bierzemy: powiela promocję (README.md, 2026-09-23).
KNOWN = ["epidemic", "promotion"]
# Historia sprzedaży serii, zawsze z opóźnieniem >= LAG dni.
LAGS = ["lag_28", "roll7_lag28", "roll28_lag28"]
FEATURES = CATEGORICAL + KNOWN + LAGS

BASE_PARAMS = {
    "learning_rate": 0.05,
    "random_state": cfg.RANDOM_STATE,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
PARAM_GRID = {
    "num_leaves": [4, 7, 15, 31, 63],
    "n_estimators": [50, 100, 200, 500],
    "min_child_samples": [20, 100],
}


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Panel: jeden wiersz na (dzień, sklep, produkt), z cechami i celem `y`. Cechy ze sprzedaży bez wycieku."""
    key = [cfg.COL_STORE, cfg.COL_PRODUCT]
    df = raw.sort_values(key + [cfg.COL_DATE]).reset_index(drop=True)

    df["known_sales"] = df[cfg.COL_TARGET].where(df[cfg.COL_DATE] <= cfg.TRAIN_END)  # egzamin -> NaN
    df["lag_28"] = df.groupby(key)["known_sales"].shift(LAG)
    lagged = df.groupby(key)["lag_28"]
    df["roll7_lag28"] = lagged.transform(lambda s: s.rolling(7).mean())
    df["roll28_lag28"] = lagged.transform(lambda s: s.rolling(28).mean())

    panel = df.rename(
        columns={
            cfg.COL_DATE: "date", cfg.COL_STORE: "store", cfg.COL_PRODUCT: "product",
            cfg.COL_CATEGORY: "category", cfg.COL_REGION: "region", cfg.COL_WEATHER: "weather",
            cfg.COL_EPIDEMIC: "epidemic", cfg.COL_PROMOTION: "promotion",
            cfg.COL_TARGET: "y",
        }
    )[["date", *CATEGORICAL, *KNOWN, *LAGS, "y"]]
    for col in CATEGORICAL:  # stałe kategorie w całym panelu, żeby trening i prognoza kodowały je tak samo
        panel[col] = pd.Categorical(panel[col], categories=sorted(panel[col].unique()))
    return panel


def fit_predict(panel: pd.DataFrame, train_end: pd.Timestamp, test_dates, params: dict) -> pd.DataFrame:
    """Naucz model na wierszach do train_end i prognozuj test_dates; zwraca prognozy per wiersz."""
    train = panel[panel["date"] <= train_end].dropna(subset=LAGS)  # pierwsze ~55 dni nie ma pełnej historii
    test = panel[panel["date"].isin(test_dates)]
    model = lgb.LGBMRegressor(**BASE_PARAMS, **params)
    model.fit(train[FEATURES], train["y"])
    return test[["date", "store", "product"]].assign(pred=model.predict(test[FEATURES]))


def daily_total(pred: pd.DataFrame) -> pd.Series:
    """Suma prognoz po wszystkich seriach dla każdego dnia."""
    return pred.groupby("date")["pred"].sum().rename("units_sold")


def cv_score(panel: pd.DataFrame, y_train: pd.Series, params: dict) -> dict:
    """Średnie metryki (dla sumy dziennej) z foldów walk-forward."""
    fold_metrics = []
    for fold in walk_forward_folds(y_train.index):
        pred = daily_total(fit_predict(panel, fold.train[-1], fold.test, params))
        fold_metrics.append(compute_metrics(y_train.loc[fold.test], pred))
    return pd.DataFrame(fold_metrics).mean().to_dict()


def grid_search(panel: pd.DataFrame, y_train: pd.Series, grid: dict = PARAM_GRID) -> pd.DataFrame:
    """Oceń każdą kombinację parametrów walidacją kroczącą; wynik posortowany rosnąco po średnim MAE."""
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        rows.append({**params, **cv_score(panel, y_train, params)})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


if __name__ == "__main__":
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog

    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    print(grid_search(build_panel(raw), split.y_train).round(1).to_string())
