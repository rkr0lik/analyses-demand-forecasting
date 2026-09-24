"""Wczytanie danych i kontrola ich jakości. Sam plik CSV zostaje nietknięty."""
from typing import NamedTuple

import pandas as pd

import config as cfg

KEY = [cfg.COL_DATE, cfg.COL_STORE, cfg.COL_PRODUCT]


def load_raw(path=cfg.DATA_PATH) -> pd.DataFrame:
    """Wczytaj surowy CSV; kolumna z datą jako datetime."""
    return pd.read_csv(path, parse_dates=[cfg.COL_DATE])


def check_raw(df: pd.DataFrame) -> dict:
    """Sprawdź kompletność danych i zwróć podsumowanie. Przy problemie rzuca AssertionError."""
    n_days = df[cfg.COL_DATE].nunique()
    n_stores = df[cfg.COL_STORE].nunique()
    n_products = df[cfg.COL_PRODUCT].nunique()
    first, last = df[cfg.COL_DATE].min(), df[cfg.COL_DATE].max()

    assert df.isna().sum().sum() == 0, "w danych są braki"
    assert not df.duplicated(KEY).any(), "duplikaty (data, sklep, produkt)"
    assert last == cfg.TEST_END, f"ostatnia data {last.date()}, oczekiwano {cfg.TEST_END.date()}"
    assert n_days == (last - first).days + 1, "w osi czasu brakuje dni"
    assert len(df) == n_days * n_stores * n_products, "nie każda seria ma wiersz każdego dnia"

    return {
        "wiersze": len(df),
        "dni": n_days,
        "sklepy": n_stores,
        "produkty": n_products,
        "od": first.date(),
        "do": last.date(),
        "dni_treningowe": df.loc[df[cfg.COL_DATE] <= cfg.TRAIN_END, cfg.COL_DATE].nunique(),
        "dni_egzaminacyjne": df.loc[df[cfg.COL_DATE] >= cfg.TEST_START, cfg.COL_DATE].nunique(),
    }


def daily_sales(df: pd.DataFrame) -> pd.Series:
    """Łączna dzienna sprzedaż (Units Sold) po wszystkich sklepach i produktach; indeks = data."""
    sales = df.groupby(cfg.COL_DATE)[cfg.COL_TARGET].sum().asfreq("D")
    assert sales.notna().all(), "w dziennej sprzedaży brakuje dni"
    assert sales.sum() == df[cfg.COL_TARGET].sum(), "suma po agregacji różni się od sumy surowej"
    return sales.rename("units_sold")


class Split(NamedTuple):
    """Podział na trening (do TRAIN_END) i zbiór egzaminacyjny (od TEST_START do TEST_END)."""

    y_train: pd.Series
    y_test: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame


def split_train_test(sales: pd.Series, exog: pd.DataFrame) -> Split:
    """Podziel dzienną sprzedaż i zmienne zewnętrzne według dat z config.py."""
    assert sales.index.equals(exog.index), "sprzedaż i zmienne mają różne osie dat"
    split = Split(
        y_train=sales[: cfg.TRAIN_END],
        y_test=sales[cfg.TEST_START : cfg.TEST_END],
        X_train=exog[: cfg.TRAIN_END],
        X_test=exog[cfg.TEST_START : cfg.TEST_END],
    )
    train_days = (cfg.TRAIN_END - sales.index.min()).days + 1
    assert len(split.y_train) == train_days, "trening ma luki lub złą długość"
    assert len(split.y_test) == cfg.HORIZON, f"egzamin ma {len(split.y_test)} dni, oczekiwano {cfg.HORIZON}"
    assert split.y_train.index.max() < split.y_test.index.min(), "trening i egzamin nachodzą na siebie"
    assert split.y_test.index.min() - split.y_train.index.max() == pd.Timedelta(days=1), "przerwa między treningiem a egzaminem"
    assert split.X_train.index.equals(split.y_train.index) and split.X_test.index.equals(split.y_test.index)
    return split


if __name__ == "__main__":
    from src.features import daily_exog

    raw = load_raw()
    for name, value in check_raw(raw).items():
        print(f"{name}: {value}")
    sales = daily_sales(raw)
    print(f"\nagregacja: {len(sales)} dni, suma {sales.sum():,}, surowa suma {raw[cfg.COL_TARGET].sum():,}")
    print(f"średnia dzienna (cały okres): {sales.mean():.0f}")
    print(f"średnia dzienna (trening):    {sales[: cfg.TRAIN_END].mean():.0f}")
    print(f"min / max dnia: {sales.min()} / {sales.max()}")

    s = split_train_test(sales, daily_exog(raw))
    print(f"\ntrening: {len(s.y_train)} dni ({s.y_train.index.min().date()} – {s.y_train.index.max().date()})")
    print(f"egzamin: {len(s.y_test)} dni ({s.y_test.index.min().date()} – {s.y_test.index.max().date()})")
