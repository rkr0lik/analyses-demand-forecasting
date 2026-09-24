"""Bonus: prognozy dla najlepiej sprzedających się produktów z globalnego LightGBM.

Produkty wybieramy wyłącznie na danych treningowych. Prognoza produktu to suma prognoz jego serii
(sklep × produkt) po wszystkich sklepach, więc produkty sumują się do prognozy całości.
"""
import pandas as pd

import config as cfg
from src.baselines import all_baselines
from src.compare import comparison_table

TOP_N = 3


def product_daily_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Dzienna sprzedaż każdego produktu (suma po sklepach): indeks = data, kolumny = produkty."""
    return raw.pivot_table(index=cfg.COL_DATE, columns=cfg.COL_PRODUCT, values=cfg.COL_TARGET, aggfunc="sum").asfreq("D")


def mean_daily_sales_by_product(raw_train: pd.DataFrame) -> pd.Series:
    """Średnia dzienna sprzedaż produktu (suma po sklepach); przekaż tylko dane treningowe."""
    return product_daily_sales(raw_train).mean()


def top_products(raw: pd.DataFrame, n: int = TOP_N) -> list[str]:
    """N produktów o najwyższej średniej dziennej sprzedaży w okresie treningowym (sprzedaż egzaminacyjna pominięta)."""
    train = raw[raw[cfg.COL_DATE] <= cfg.TRAIN_END]
    return mean_daily_sales_by_product(train).nlargest(n).index.tolist()


def ranking_comparison(raw: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Ranking produktów liczony z treningu i z całości danych, żeby pokazać, że wybór zależy od okresu.

    Zadanie nie precyzuje okresu. Liczymy z treningu, bo okres egzaminacyjny nie może wpływać na żadną decyzję,
    ale różnice na pozycjach 2-4 są rzędu jednej sztuki dziennie, więc wybór jest praktycznie remisem.
    """
    train = mean_daily_sales_by_product(raw[raw[cfg.COL_DATE] <= cfg.TRAIN_END])
    whole = mean_daily_sales_by_product(raw)
    table = pd.DataFrame({"średnia z treningu": train, "średnia z całości": whole})
    table["pozycja (trening)"] = table["średnia z treningu"].rank(ascending=False).astype(int)
    table["pozycja (całość)"] = table["średnia z całości"].rank(ascending=False).astype(int)
    top = table.nsmallest(n, "pozycja (trening)").index.union(table.nsmallest(n, "pozycja (całość)").index)
    return table.loc[top].sort_values("pozycja (trening)").rename_axis("produkt")


def load_series_forecasts() -> pd.DataFrame:
    """Prognozy egzaminacyjne per seria (date, store, product, pred) zapisane przez `python -m src.run_models`."""
    path = cfg.OUTPUT_DIR / "forecasts_lgbm_series.csv"
    if not path.exists():
        raise FileNotFoundError(f"brak {path}; uruchom najpierw: python -m src.run_models")
    return pd.read_csv(path, parse_dates=["date"])


def product_forecasts(series_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Prognozy produktów: suma po sklepach; indeks = data, kolumny = produkty."""
    return series_forecasts.pivot_table(index="date", columns="product", values="pred", aggfunc="sum")


def product_candidates(sales: pd.DataFrame, product: str, forecasts: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Rzeczywista sprzedaż egzaminacyjna produktu oraz prognozy: baseline'y z jego treningu i LightGBM."""
    exam_days = pd.date_range(cfg.TEST_START, cfg.TEST_END)
    y_train, y_test = sales.loc[: cfg.TRAIN_END, product], sales.loc[exam_days, product]
    return y_test, all_baselines(y_train).assign(LightGBM=forecasts.loc[exam_days, product])


def top_products_comparison(
    raw: pd.DataFrame, products: list[str], forecasts: pd.DataFrame, epidemic: pd.Series
) -> pd.DataFrame:
    """Metryki na egzaminie dla każdego produktu: LightGBM i baseline'y liczone z treningu tego produktu.

    Wiersze: (produkt, model). Kolumny jak w tabeli zbiorczej: metryki razem oraz osobno dla epidemii i bez.
    """
    sales = product_daily_sales(raw)
    tables = {}
    for product in products:
        y_test, candidates = product_candidates(sales, product, forecasts)
        tables[product] = comparison_table(y_test, candidates, epidemic).drop(columns="grupa")
    return pd.concat(tables, names=["produkt", "model"])


def top_products_forecasts(raw: pd.DataFrame, products: list[str], forecasts: pd.DataFrame) -> pd.DataFrame:
    """Prognozy egzaminacyjne w formie długiej: date, product, actual oraz kolumna na każdy model."""
    sales = product_daily_sales(raw)
    frames = []
    for product in products:
        y_test, candidates = product_candidates(sales, product, forecasts)
        frames.append(candidates.assign(product=product, actual=y_test))
    long = pd.concat(frames).rename_axis("date").reset_index()
    return long[["date", "product", "actual", *candidates.columns]]


if __name__ == "__main__":
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog

    raw = load_raw()
    top = top_products(raw)
    forecasts = product_forecasts(load_series_forecasts())
    epidemic = split_train_test(daily_sales(raw), daily_exog(raw)).X_test["epidemic"]

    print(f"top {TOP_N} produktów (trening): {top}")
    print(f"top {TOP_N} produktów (całość danych, tylko dla porównania): "
          f"{ranking_comparison(raw).nsmallest(TOP_N, 'pozycja (całość)').index.tolist()}\n")
    ranking = ranking_comparison(raw)
    print(ranking.round(1).to_string())
    print()
    table = top_products_comparison(raw, top, forecasts, epidemic)
    pd.set_option("display.width", 200)
    print(table.astype(float).round(1).to_string())

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(cfg.OUTPUT_DIR / "comparison_top_products.csv")
    top_products_forecasts(raw, top, forecasts).to_csv(cfg.OUTPUT_DIR / "forecasts_top_products.csv", index=False)
    ranking.to_csv(cfg.OUTPUT_DIR / "top_products_ranking.csv")
    print("\nzapisano: comparison_top_products.csv, forecasts_top_products.csv, top_products_ranking.csv")
