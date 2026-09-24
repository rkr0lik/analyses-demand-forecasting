"""Zmienne zewnętrzne (egzogeniczne) w ujęciu dziennym, do modeli działających na sumie sprzedaży."""
import numpy as np
import pandas as pd

import config as cfg

# Pogoda jest ustalana per sklep, więc w danych dziennych to udział sklepów z daną pogodą.
# Cloudy zostaje poza zestawem jako kategoria odniesienia: cztery udziały sumują się do 1,
# a modele liniowe nie lubią cech, które razem dają stałą (współliniowość).
WEATHER_SHARES = {"Sunny": "sunny_share", "Rainy": "rainy_share", "Snowy": "snowy_share"}

EXOG_COLUMNS = ["epidemic", "promo_share", *WEATHER_SHARES.values()]

# Cykl roczny: sprzedaż ma silną sezonowość roczną (audyt 2026-09-22, README.md). Opisujemy ją falami
# sin/cos dnia roku — kilka gładkich kolumn zamiast jedenastu wskaźników miesiąca. Fale zależą wyłącznie
# od daty, więc są znane z dowolnym wyprzedzeniem i nie mogą wnieść wycieku danych.
SEASONAL_WAVES = 3  # domyślna liczba par fal (wariant wybrany walidacją kroczącą dla Ridge)


def year_waves(index: pd.DatetimeIndex, waves: int = SEASONAL_WAVES) -> pd.DataFrame:
    """Cykl roczny jako `waves` par fal sin/cos dnia roku."""
    day = index.dayofyear.to_numpy()
    return pd.DataFrame(
        {f"{f}{i}": getattr(np, f)(2 * np.pi * i * day / 365.25) for i in range(1, waves + 1) for f in ("sin", "cos")},
        index=index,
    )


def daily_exog(df: pd.DataFrame) -> pd.DataFrame:
    """Jeden wiersz na dzień: epidemia, udział produktów w promocji, średni rabat, udziały pogody.

    To zmienne znane (lub planowane) niezależnie od sprzedaży. Dla dni egzaminacyjnych podajemy
    prawdziwe wartości, czyli wariant oracle.
    """
    by_day = df.groupby(cfg.COL_DATE)
    assert (by_day[cfg.COL_EPIDEMIC].nunique() == 1).all(), "Epidemic nie jest jedna na dzień"

    exog = pd.DataFrame(
        {
            "epidemic": by_day[cfg.COL_EPIDEMIC].first(),  # flaga dnia: 1 = epidemia
            # Jaka część produktów jest danego dnia w promocji. Rabatu NIE bierzemy osobno: w tych danych
            # jest funkcją promocji (bez promocji 0/5/10%, z promocją 10/15/20/25%), więc tylko powielał
            # tę samą informację i wprowadzał współliniowość (README.md, 2026-09-23).
            "promo_share": by_day[cfg.COL_PROMOTION].mean(),
        }
    )
    for weather, name in WEATHER_SHARES.items():
        exog[name] = (df[cfg.COL_WEATHER] == weather).groupby(df[cfg.COL_DATE]).mean()
    return exog[EXOG_COLUMNS].asfreq("D")


if __name__ == "__main__":
    from src.data import daily_sales, load_raw

    raw = load_raw()
    exog = daily_exog(raw)
    sales = daily_sales(raw)

    assert exog.index.equals(sales.index), "oś dat zmiennych różni się od osi sprzedaży"
    assert exog.notna().all().all(), "w zmiennych zewnętrznych są braki"
    assert exog["epidemic"].isin([0, 1]).all()
    assert exog.ge(0).all().all() and exog.le(1).all().all(), "wszystkie cechy to udziały z zakresu [0, 1]"

    print(f"kształt: {exog.shape}, dni epidemii: {int(exog['epidemic'].sum())}")
    print(f"epidemia w okresie egzaminacyjnym: {int(exog.loc[cfg.TEST_START:, 'epidemic'].sum())} dni")
    print(exog.describe().round(3).T[["mean", "min", "max"]])
