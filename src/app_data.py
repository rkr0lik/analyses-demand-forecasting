"""Dane i logika widoków aplikacji. Wczytuje gotowe pliki z outputs/ i składa z nich widoki; niczego nie uczy.

Aplikacja (app/streamlit_app.py) woła tylko te funkcje oraz wspólną funkcję metryk, więc nie potrzebuje
pliku z surowymi danymi.
"""
from typing import NamedTuple

import pandas as pd

import config as cfg
from src.compare import forecasts_from_outputs
from src.intervals import bands
from src.metrics import compute_metrics, metrics_table

TOTAL = "Wszystkie produkty"  # suma po wszystkich sklepach i produktach
BASELINE_MAIN = f"średnia {cfg.MEAN_WINDOW} dni"  # baseline, z którym porównujemy się w KPI
BASELINES = ["naive", "seasonal naive", BASELINE_MAIN]
TOTAL_MODELS = ["ARIMAX", "Ridge", "LightGBM", "Ensemble zwykła średnia", "Ensemble ważona średnia"]  # mają pasmo i scenariusze
PRODUCT_MODELS = ["LightGBM"]
EXAM_DAYS = pd.date_range(cfg.TEST_START, cfg.TEST_END)


def real_epidemic_days(daily: pd.DataFrame) -> int:
    """Ile dni epidemii faktycznie było na początku okresu prognozy (liczone z danych, nie wpisane na sztywno)."""
    flags = daily.loc[EXAM_DAYS, "epidemic"].to_numpy()
    assert set(flags) <= {0, 1}, "flaga epidemii musi być 0/1"
    first_zero = flags.argmin() if not flags.all() else len(flags)
    assert not flags[first_zero:].any(), "epidemia w okresie prognozy nie jest ciągła od początku"
    return int(first_zero)


class Outputs(NamedTuple):
    daily: pd.DataFrame  # indeks: data; kolumny units_sold, epidemic
    product_sales: pd.DataFrame  # indeks: data; kolumny: produkty
    forecasts: pd.DataFrame  # prognozy egzaminacyjne 9 modeli
    quantiles: pd.DataFrame  # przesuw pasma 80% dla 5 modeli
    scenarios: pd.DataFrame  # epidemic_days, date, model, forecast
    product_scenarios: pd.DataFrame  # epidemic_days, date, product, LightGBM
    top_products: pd.DataFrame  # date, product, actual + prognozy produktów
    comparison: pd.DataFrame  # tabela zbiorcza 9 modeli na egzaminie


class ViewData(NamedTuple):
    view: str
    model: str
    epidemic_days: int
    actual: pd.Series  # rzeczywista sprzedaż, cała historia
    epidemic: pd.Series  # flaga epidemii, cała historia
    forecast: pd.Series  # prognoza wybranego modelu na dni egzaminacyjne
    baselines: pd.DataFrame  # prognozy baseline'ów na dni egzaminacyjne
    band: pd.DataFrame | None  # kolumny lower, upper (tylko widok zbiorczy)


def _read(name: str, script: str, **kwargs) -> pd.DataFrame:
    path = cfg.OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"brak {path}; uruchom najpierw: python -m {script}")
    return pd.read_csv(path, **kwargs)


def load_outputs() -> Outputs:
    """Wczytaj wszystkie pliki potrzebne aplikacji (tworzą je skrypty z src/)."""
    daily = _read("daily_sales.csv", "src.export_actuals", index_col="date", parse_dates=True)
    forecasts = forecasts_from_outputs(daily["units_sold"][: cfg.TRAIN_END], EXAM_DAYS)
    return Outputs(
        daily=daily,
        product_sales=_read("product_daily_sales.csv", "src.export_actuals", index_col="date", parse_dates=True),
        forecasts=forecasts,
        quantiles=_read("interval_quantiles.csv", "src.intervals", index_col="model"),
        scenarios=_read("scenarios.csv", "src.scenarios", parse_dates=["date"]),
        product_scenarios=_read("scenarios_top_products.csv", "src.scenarios", parse_dates=["date"]),
        top_products=_read("forecasts_top_products.csv", "src.bonus", parse_dates=["date"]),
        comparison=_read("comparison.csv", "src.compare", index_col="model"),
    )


def views(outputs: Outputs) -> list[str]:
    """Dostępne widoki: całość i produkty bonusowe."""
    return [TOTAL, *outputs.top_products["product"].unique().tolist()]


def models_for(view: str) -> list[str]:
    return TOTAL_MODELS if view == TOTAL else PRODUCT_MODELS


def build_view(outputs: Outputs, view: str, model: str, epidemic_days: int) -> ViewData:
    """Złóż dane wykresu i tabeli dla wybranego widoku, modelu i scenariusza epidemii."""
    assert model in models_for(view), f"model {model} niedostępny w widoku {view}"
    epidemic = outputs.daily["epidemic"]
    if view == TOTAL:
        part = outputs.scenarios[(outputs.scenarios["model"] == model) & (outputs.scenarios["epidemic_days"] == epidemic_days)]
        forecast = part.set_index("date")["forecast"]
        band = bands(forecast.rename(model).to_frame(), outputs.quantiles.loc[[model]]).set_index("date")[["lower", "upper"]]
        return ViewData(view, model, epidemic_days, outputs.daily["units_sold"], epidemic, forecast, outputs.forecasts[BASELINES], band)

    part = outputs.product_scenarios[(outputs.product_scenarios["product"] == view) & (outputs.product_scenarios["epidemic_days"] == epidemic_days)]
    exam = outputs.top_products[outputs.top_products["product"] == view].set_index("date")
    return ViewData(view, model, epidemic_days, outputs.product_sales[view], epidemic, part.set_index("date")["LightGBM"], exam[BASELINES], None)


def exam_days_in(data: ViewData, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Dni egzaminacyjne mieszczące się w wybranym zakresie dat (tylko one mają prognozę i rzeczywistość)."""
    return data.forecast.index[(data.forecast.index >= start) & (data.forecast.index <= end)]


def metrics_for_range(data: ViewData, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Metryki wybranego modelu i baseline'ów na dniach egzaminacyjnych z zakresu; None, gdy zakres ich nie obejmuje.

    Kolumna 'MAE vs średnia 28 dni' to zmiana MAE względem głównego baseline'u w procentach (ujemna = lepiej).
    """
    days = exam_days_in(data, start, end)
    if days.empty:
        return None
    y = data.actual.loc[days]
    candidates = {data.model: data.forecast.loc[days], **{name: data.baselines.loc[days, name] for name in BASELINES}}
    table = pd.DataFrame({name: {**compute_metrics(y, pred), "dni": len(days)} for name, pred in candidates.items()}).T
    table[f"MAE vs {BASELINE_MAIN}"] = (table["MAE"] / table.loc[BASELINE_MAIN, "MAE"] - 1) * 100
    return table


def epidemic_split(data: ViewData, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Metryki wybranego modelu osobno dla dni z epidemią i bez (wspólna funkcja metryk)."""
    days = exam_days_in(data, start, end)
    if days.empty:
        return None
    return metrics_table(data.actual.loc[days], data.forecast.loc[days], data.epidemic.loc[days])


def epidemic_runs(epidemic: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Ciągłe okresy epidemii (pierwszy i ostatni dzień każdego), do zacieniowania na wykresie."""
    flag = epidemic.astype(bool)
    run_id = (flag != flag.shift()).cumsum()
    return [(days.index[0], days.index[-1]) for _, days in flag[flag].groupby(run_id[flag])]
