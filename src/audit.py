"""Kontrola poprawności: przelicza wyniki jeszcze raz i szuka słabych punktów analizy.

Modeli nie zmienia. Wyniki trafiają do outputs/, skąd biorą je tabele w dokumentacji. Sprawdzamy:

1. czy metryki w tabeli porównawczej dają się odtworzyć z zapisanych prognoz,
2. jak wygląda błąd modelu w poszczególnych miesiącach (sezonowość roczna),
3. ile daje cykl roczny w Ridge w walidacji kroczącej i poza nią,
4. czy różnice między modelami na egzaminie są istotne statystycznie,
5. kilka pojedynczych liczb, które cytuje dokumentacja.

Użycie: python -m src.audit
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller

import config as cfg
from src import ridge
from src.compare import SOURCES
from src.data import daily_sales, load_raw, split_train_test
from src.features import EXOG_COLUMNS, daily_exog
from src.metrics import compute_metrics
from src.validation import walk_forward_folds

MONTH_NAMES = ["styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
               "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień"]
SEASONAL_WAVES = 2  # wariant porównawczy w tabeli poniżej (Ridge używa cfg.RIDGE_YEAR_WAVES)


def recomputed_metrics(split) -> pd.DataFrame:
    """Metryki policzone od nowa z zapisanych prognoz, obok tych z tabeli porównawczej."""
    saved = pd.read_csv(cfg.OUTPUT_DIR / "comparison.csv", index_col="model")
    rows = {}
    for filename, (columns, _) in SOURCES.items():
        forecasts = pd.read_csv(cfg.OUTPUT_DIR / filename, index_col="date", parse_dates=True)
        for model in columns:
            fresh = compute_metrics(split.y_test, forecasts[model])
            rows[model] = {
                **{f"{k} (przeliczone)": v for k, v in fresh.items()},
                "największa różnica": max(abs(fresh[k] - saved.loc[model, k]) for k in fresh),
            }
    return pd.DataFrame(rows).T.rename_axis("model")


def month_effects(y_train: pd.Series, X_train: pd.DataFrame) -> pd.DataFrame:
    """Średni błąd regresji treningowej w poszczególnych miesiącach, czyli czego zmienne nie tłumaczą."""
    fit = sm.OLS(y_train, sm.add_constant(X_train[EXOG_COLUMNS])).fit()
    resid = y_train - fit.predict(sm.add_constant(X_train[EXOG_COLUMNS]))
    grouped = resid.groupby(resid.index.month)
    table = pd.DataFrame({"efekt": grouped.mean(), "dni": grouped.count(), "rozrzut": grouped.std()})
    table["blad_standardowy"] = table["rozrzut"] / np.sqrt(table["dni"])
    table.index = [MONTH_NAMES[m - 1] for m in table.index]
    return table.rename_axis("miesiąc")[["efekt", "blad_standardowy", "dni"]]


def seasonality_gain(y_train: pd.Series, X_train: pd.DataFrame, alphas=(1, 3, 10, 30, 100, 300)) -> pd.DataFrame:
    """Ile cykl roczny daje Ridge w walidacji kroczącej na treningu.

    Porównuje różne liczby par fal, w tym waves=0 (bez cyklu). Model używa cfg.RIDGE_YEAR_WAVES.
    """
    def cv(waves):
        best = None
        for alpha in alphas:
            folds = [compute_metrics(y_train.loc[f.test],
                                     ridge.fit_forecast(y_train[: f.train[-1]], X_train[: f.train[-1]],
                                                        X_train.loc[f.test], alpha, waves))
                     for f in walk_forward_folds(y_train.index)]
            mean = pd.DataFrame(folds).mean()
            if best is None or mean["MAE"] < best[0]["MAE"]:
                best = (mean, alpha)
        return {"MAE": best[0]["MAE"], "Bias": best[0]["Bias"], "alpha": best[1]}

    rows = {"bez sezonowości rocznej": cv(0)}
    for waves in (1, SEASONAL_WAVES, 3):
        label = f"z sezonowością roczną ({waves} {'para' if waves == 1 else 'pary'} fal)"
        if waves == cfg.RIDGE_YEAR_WAVES:
            label += " — obecny model"
        rows[label] = cv(waves)
    return pd.DataFrame(rows).T.rename_axis("wariant")


def year_cycle_check(y_train, X_train, split) -> pd.DataFrame:
    """Ridge z cyklem rocznym i bez niego: na egzaminie oraz w kontrolnym styczniu rok wcześniej.

    Styczeń 2023 leży w okresie treningowym, więc ten sam układ dat co egzamin można przećwiczyć
    rok wcześniej, nie ruszając zbioru egzaminacyjnego. Poprawa widoczna w walidacji
    (tabela wyżej) na styczniu się nie pojawia.
    """
    january_end = cfg.TRAIN_END - pd.DateOffset(years=1)  # 2.01.2023
    january = pd.date_range(january_end + pd.Timedelta(days=1), periods=cfg.HORIZON)
    rows = {}
    for label, waves in (("bez cyklu rocznego", 0), (f"z cyklem rocznym ({cfg.RIDGE_YEAR_WAVES} pary fal)", cfg.RIDGE_YEAR_WAVES)):
        exam = ridge.fit_forecast(split.y_train, split.X_train, split.X_test, cfg.RIDGE_ALPHA, waves)
        jan = ridge.fit_forecast(y_train[:january_end], X_train[:january_end], X_train.loc[january], cfg.RIDGE_ALPHA, waves)
        rows[label] = {
            "MAE egzamin": compute_metrics(split.y_test, exam)["MAE"],
            "Bias egzamin": compute_metrics(split.y_test, exam)["Bias"],
            "MAE styczeń rok wcześniej": compute_metrics(y_train.loc[january], jan)["MAE"],
        }
    return pd.DataFrame(rows).T.rename_axis("wariant Ridge")


def extra_checks(raw: pd.DataFrame, split, months: pd.DataFrame) -> pd.DataFrame:
    """Pojedyncze liczby, które cytuje dokumentacja."""
    daily = raw.groupby(cfg.COL_DATE)[["Demand", cfg.COL_TARGET]].sum()
    fit = sm.OLS(split.y_train, sm.add_constant(split.X_train[EXOG_COLUMNS])).fit()
    resid = split.y_train - fit.predict(sm.add_constant(split.X_train[EXOG_COLUMNS]))
    adf_p = adfuller(resid, autolag="AIC")[1]

    bands = pd.read_csv(cfg.OUTPUT_DIR / "intervals.csv", parse_dates=["date"])
    below = above = 0
    for _, part in bands.groupby("model"):
        actual = split.y_test.loc[part["date"]].to_numpy()
        below += int((actual < part["lower"].to_numpy()).sum())
        above += int((actual > part["upper"].to_numpy()).sum())

    spread = months["efekt"].max() - months["efekt"].min()
    rows = {
        "korelacja kolumny Demand ze sprzedażą": daily["Demand"].corr(daily[cfg.COL_TARGET]),
        "rozstęp efektu miesiąca (szt./dzień)": spread,
        "rozstęp efektu miesiąca jako % sprzedaży": spread / split.y_train.mean() * 100,
        "test stacjonarności reszt regresji (p)": adf_p,
        "dni poniżej pasma (wszystkie modele razem)": below,
        "dni powyżej pasma (wszystkie modele razem)": above,
    }
    return pd.DataFrame({"wartość": rows}).rename_axis("sprawdzenie")


def difference_tests(split) -> pd.DataFrame:
    """Test parowany: czy różnice błędów między modelami na 28 dniach egzaminu są istotne."""
    forecasts = {}
    for filename, (columns, _) in SOURCES.items():
        frame = pd.read_csv(cfg.OUTPUT_DIR / filename, index_col="date", parse_dates=True)
        forecasts.update({model: frame[model] for model in columns})
    errors = {m: (f - split.y_test).abs().to_numpy() for m, f in forecasts.items()}
    order = sorted(errors, key=lambda m: errors[m].mean())
    best = order[0]
    rows = {}
    for model in order[1:]:
        p_value = stats.ttest_rel(errors[best], errors[model]).pvalue
        rows[f"{best} vs {model}"] = {
            "różnica MAE": errors[model].mean() - errors[best].mean(),
            "p": p_value,
            "istotna": "tak" if p_value < 0.05 else "nie",
        }
    return pd.DataFrame(rows).T.rename_axis("porównanie")


if __name__ == "__main__":
    raw = load_raw()
    split = split_train_test(daily_sales(raw), daily_exog(raw))
    pd.set_option("display.width", 200)

    metrics = recomputed_metrics(split)
    months = month_effects(split.y_train, split.X_train)
    gain = seasonality_gain(split.y_train, split.X_train)
    cycle = year_cycle_check(split.y_train, split.X_train, split)
    tests = difference_tests(split)
    checks = extra_checks(raw, split, months)

    print("1. Metryki przeliczone od nowa z zapisanych prognoz")
    print(metrics.round(4).to_string())
    print(f"\nNajwiększa różnica wobec tabeli porównawczej: {metrics['największa różnica'].max():.2e}")

    print("\n2. Średni błąd regresji w poszczególnych miesiącach (trening)")
    print(months.round(0).to_string())

    print("\n3. Czy sezonowość roczna pomaga (walidacja krocząca, egzamin nietknięty)")
    print(gain.round(1).to_string())

    print("\n3b. Co cykl roczny dał Ridge poza walidacją (egzamin i kontrolny styczeń rok wcześniej)")
    print(cycle.round(1).to_string())

    print("\n4. Istotność różnic między modelami na egzaminie")
    print(tests.round(3).to_string())

    print("\n5. Pozostałe sprawdzenia")
    print(checks.round(4).to_string())

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    months.to_csv(cfg.OUTPUT_DIR / "audit_months.csv")
    gain.to_csv(cfg.OUTPUT_DIR / "audit_seasonality.csv")
    cycle.to_csv(cfg.OUTPUT_DIR / "audit_year_cycle.csv")
    tests.to_csv(cfg.OUTPUT_DIR / "audit_tests.csv")
    checks.to_csv(cfg.OUTPUT_DIR / "audit_checks.csv")
    pd.DataFrame({"największa różnica metryk": [metrics["największa różnica"].max()]}).to_csv(
        cfg.OUTPUT_DIR / "audit_metrics.csv", index=False
    )
    print("\nzapisano: audit_months.csv, audit_seasonality.csv, audit_year_cycle.csv, audit_tests.csv, audit_checks.csv, audit_metrics.csv")
