"""Eksploracja danych treningowych (do 2.01.2024): efekt epidemii, rytm tygodniowy, regresja, cena i ranking produktów.

Okres egzaminacyjny jest tu pominięty.
"""
import pandas as pd
import statsmodels.api as sm

import config as cfg
from src.bonus import mean_daily_sales_by_product
from src.data import daily_sales, load_raw
from src.features import EXOG_COLUMNS, daily_exog


def ols(y: pd.Series, X: pd.DataFrame):
    return sm.OLS(y, sm.add_constant(X)).fit()


def epidemic_effect(sales: pd.Series, exog: pd.DataFrame) -> None:
    means = sales.groupby(exog["epidemic"]).mean()
    print(f"[epidemia] bez: {means[0]:.0f}, z: {means[1]:.0f}, zmiana: {means[1] / means[0] - 1:+.1%}")


def weekday_pattern(sales: pd.Series) -> None:
    by_dow = sales.groupby(sales.index.dayofweek).mean()
    spread = (by_dow.max() - by_dow.min()) / sales.mean()
    print(f"[dzień tygodnia] średnie: {by_dow.round(0).astype(int).tolist()} (pn..nd), rozstęp/średnia: {spread:.1%}")
    # Epidemia może zamazać rytm tygodniowy, więc liczymy go też bez dni epidemii.


def weekday_pattern_no_epidemic(sales: pd.Series, exog: pd.DataFrame) -> None:
    calm = sales[exog["epidemic"] == 0]
    by_dow = calm.groupby(calm.index.dayofweek).mean()
    print(f"[dzień tygodnia, bez epidemii] rozstęp/średnia: {(by_dow.max() - by_dow.min()) / calm.mean():.1%}")


def regression_all(sales: pd.Series, exog: pd.DataFrame) -> None:
    fit = ols(sales, exog[EXOG_COLUMNS])
    print(f"[regresja liniowa, wszystkie zmienne] R2 = {fit.rsquared:.3f}")
    print(fit.params.round(1).to_string())
    only_epi = ols(sales, exog[["epidemic"]]).rsquared
    print(f"[R2 tylko z epidemią] {only_epi:.3f}")


def price_vs_epidemic(raw_train: pd.DataFrame, sales: pd.Series, exog: pd.DataFrame) -> None:
    price = raw_train.groupby(cfg.COL_DATE)["Price"].mean().asfreq("D")
    by_epi = price.groupby(exog["epidemic"]).mean()
    print(f"[cena] średnia bez epidemii: {by_epi[0]:.2f}, w epidemii: {by_epi[1]:.2f}")
    print(f"[cena] korelacja z epidemią: {price.corr(exog['epidemic']):.2f}, ze sprzedażą: {price.corr(sales):.2f}")
    base = ols(sales, exog[EXOG_COLUMNS])
    with_price = ols(sales, exog[EXOG_COLUMNS].assign(price=price))
    print(f"[cena] R2 bez ceny: {base.rsquared:.4f}, z ceną: {with_price.rsquared:.4f}, "
          f"współczynnik ceny: {with_price.params['price']:.1f} (p={with_price.pvalues['price']:.3f})")


def top_products(raw_train: pd.DataFrame, n: int = 3) -> None:
    top = mean_daily_sales_by_product(raw_train).nlargest(n)
    print(f"[top {n} produktów wg średniej dziennej sprzedaży, trening]")
    print(top.round(0).astype(int).to_string())


if __name__ == "__main__":
    raw = load_raw()
    raw_train = raw[raw[cfg.COL_DATE] <= cfg.TRAIN_END]
    sales = daily_sales(raw)[: cfg.TRAIN_END]
    exog = daily_exog(raw)[: cfg.TRAIN_END]
    print(f"dane treningowe: {len(sales)} dni ({sales.index.min().date()} – {sales.index.max().date()})\n")

    epidemic_effect(sales, exog)
    weekday_pattern(sales)
    weekday_pattern_no_epidemic(sales, exog)
    print()
    regression_all(sales, exog)
    print()
    price_vs_epidemic(raw_train, sales, exog)
    print()
    top_products(raw_train)
