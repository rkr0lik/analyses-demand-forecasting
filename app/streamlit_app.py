"""Aplikacja Streamlit: prognoza dziennej sprzedaży na 28 dni, w stylu raportu Power BI.

Uruchomienie z katalogu głównego: streamlit run app/streamlit_app.py
Aplikacja tylko wczytuje gotowe pliki z outputs/ (src/app_data.py) i wywołuje wspólną funkcję metryk; niczego nie uczy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # żeby `import config` i `import src` działały spod streamlit run

import pandas as pd
import streamlit as st

import config as cfg
from app.charts import build_figure
from src.app_data import (
    BASELINE_MAIN,
    build_view,
    epidemic_split,
    load_outputs,
    metrics_for_range,
    models_for,
    real_epidemic_days,
    views,
)

NBSP = " "


def fmt_int(value: float, signed: bool = False) -> str:
    """Liczba całkowita z twardą spacją jako separatorem tysięcy (np. 2 460)."""
    return f"{value:{'+' if signed else ''},.0f}".replace(",", NBSP).replace("-", "−")


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%".replace(".", ",")


def fmt_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%d.%m.%Y")


@st.cache_data
def get_outputs():
    return load_outputs()


def kpi_card(key: str, value: str, label: str) -> None:
    with st.container(key=f"card_{key}"):
        st.markdown(f'<div class="kpi-value">{value}</div><div class="kpi-label">{label}</div>', unsafe_allow_html=True)


def card_title(text: str, sub: str | None = None) -> None:
    st.markdown(f'<div class="card-title">{text}</div>' + (f'<div class="card-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


st.set_page_config(page_title="Prognoza sprzedaży na 28 dni", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

try:
    outputs = get_outputs()
except FileNotFoundError as error:
    st.error(f"Brakuje plików z wynikami. {error}")
    st.stop()

real_epidemic = real_epidemic_days(outputs.daily)  # liczone z danych, nie wpisane na sztywno

st.markdown(
    f'<div class="pbi-header"><span class="pbi-title">Prognoza dziennej sprzedaży na {cfg.HORIZON} dni</span>'
    f'<span class="pbi-tag">Wariant oracle</span>'
    f'<span class="pbi-sub">Trening do {fmt_date(cfg.TRAIN_END)} · prognoza {fmt_date(cfg.TEST_START)} – {fmt_date(cfg.TEST_END)} · '
    f"znane z góry: epidemia, promocje, rabaty, pogoda</span></div>",
    unsafe_allow_html=True,
)

# --- Pasek fragmentatorów (jeden rząd nad wszystkim, czego dotyczy) ---
slicer_view, slicer_model, slicer_range, slicer_scenario = st.columns([1.2, 1.2, 1.8, 1.8])
with slicer_view, st.container(key="card_slicer_view"):
    card_title("Widok")
    view = st.selectbox("Widok", views(outputs), key="view", label_visibility="collapsed")
with slicer_model, st.container(key="card_slicer_model"):
    card_title("Model")
    model = st.selectbox("Model", models_for(view), key=f"model_{view}", label_visibility="collapsed")
with slicer_range, st.container(key="card_slicer_range"):
    card_title("Zakres dat")
    first_day, last_day = outputs.daily.index.min().date(), cfg.TEST_END.date()
    default_start = max(first_day, (cfg.TRAIN_END - pd.Timedelta(days=55)).date())
    start_date, end_date = st.slider(
        "Zakres dat", min_value=first_day, max_value=last_day, value=(default_start, last_day), format="DD.MM.YYYY",
        key="range", label_visibility="collapsed",
    )
with slicer_scenario, st.container(key="card_slicer_scenario"):
    card_title("Scenariusz epidemii", f"Epidemia trwa jeszcze N dni od {fmt_date(cfg.TEST_START)} (rzeczywiście: {real_epidemic})")
    epidemic_days = st.slider(
        "Liczba dni epidemii w okresie prognozy", 0, cfg.HORIZON, real_epidemic, key="epidemic_days", label_visibility="collapsed"
    )

start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
data = build_view(outputs, view, model, epidemic_days)
table = metrics_for_range(data, start, end)
scenario_note = "zgodnie z rzeczywistością" if epidemic_days == real_epidemic else "scenariusz hipotetyczny"

# --- Karty KPI ---
kpis = st.columns(4)
if table is None:
    values = ["—"] * 4
else:
    row, delta = table.loc[model], table.loc[model, f"MAE vs {BASELINE_MAIN}"]
    values = [fmt_int(row["MAE"]), fmt_pct(row["MAPE"]), fmt_int(row["Bias"], signed=True), f"{'▼' if delta < 0 else '▲'} {abs(delta):.0f}%"]
labels = [
    "MAE, szt./dzień",
    "MAPE, błąd procentowy",
    "Bias, szt./dzień (plus = zawyża)",
    f"MAE względem: {BASELINE_MAIN} (▼ = mniejszy błąd)",
]
for column, key, value, label in zip(kpis, ("mae", "mape", "bias", "delta"), values, labels):
    with column:
        kpi_card(f"kpi_{key}", value, label)

if table is None:
    st.info(f"Wybrany zakres nie obejmuje dni prognozy ({fmt_date(cfg.TEST_START)} – {fmt_date(cfg.TEST_END)}). "
            "Metryki liczymy tylko dla dni, dla których jest prognoza i sprzedaż rzeczywista.")

# --- Wykres i tabela metryk ---
chart_col, table_col = st.columns([3, 2])
with chart_col, st.container(key="card_chart"):
    card_title("Sprzedaż dzienna: rzeczywista i prognoza", f"{view} · model {model} · epidemia {epidemic_days} dni ({scenario_note})")
    st.plotly_chart(build_figure(data, start, end), config={"displayModeBar": False}, key="chart")

with table_col, st.container(key="card_table"):
    card_title("Metryki: model i baseline'y", None if table is None else f"Dni prognozy w wybranym zakresie: {int(table['dni'].iloc[0])}")
    if table is not None:
        shown = table.drop(columns="dni").rename(columns={"MAPE": "MAPE (%)"}).rename_axis("Model")
        st.dataframe(
            shown, key="metrics_table",
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.0f"), "RMSE": st.column_config.NumberColumn(format="%.0f"),
                "MAPE (%)": st.column_config.NumberColumn(format="%.1f"), "Bias": st.column_config.NumberColumn(format="%+.0f"),
                f"MAE vs {BASELINE_MAIN}": st.column_config.NumberColumn(f"MAE vs {BASELINE_MAIN} (%)", format="%+.0f"),
            },
        )
        split = epidemic_split(data, start, end)
        card_title("Wybrany model osobno dla dni z epidemią i bez")
        st.dataframe(
            split.drop(columns="dni").assign(dni=split["dni"].astype(int)).rename(columns={"MAPE": "MAPE (%)"}).rename_axis("Okres"),
            key="epidemic_table",
            column_config={c: st.column_config.NumberColumn(format="%.1f" if c == "MAPE (%)" else "%.0f") for c in ("MAE", "RMSE", "MAPE (%)", "Bias")},
        )

# --- Widok tabelaryczny i opis ---
with st.expander("Widok tabelaryczny: dane wykresu"):
    days = data.forecast.index[(data.forecast.index >= start) & (data.forecast.index <= end)]
    view_table = pd.DataFrame({"Sprzedaż rzeczywista": data.actual.loc[days], f"Prognoza: {data.model}": data.forecast.loc[days]})
    if data.band is not None:
        view_table[["Pasmo 80%: dolna", "Pasmo 80%: górna"]] = data.band.loc[days, ["lower", "upper"]]
    view_table[f"Baseline: {BASELINE_MAIN}"] = data.baselines.loc[days, BASELINE_MAIN]
    view_table["Epidemia"] = data.epidemic.loc[days].map({0: "nie", 1: "tak"})
    st.dataframe(view_table.rename_axis("Data").round(0), key="chart_table")

with st.expander("Wszystkie modele na zbiorze egzaminacyjnym"):
    st.dataframe(outputs.comparison.rename_axis("Model"), key="all_models_table")

with st.expander("Jak czytać wyniki i ograniczenia"):
    st.markdown(
        f"""
- **Wariant oracle.** Model zna z góry prawdziwe wartości epidemii, promocji, rabatów i pogody dla dni prognozy ({fmt_date(cfg.TEST_START)} – {fmt_date(cfg.TEST_END)}). W praktyce epidemii i pogody z takim wyprzedzeniem nie znamy, dlatego jest suwak scenariusza epidemii.
- **Scenariusz epidemii** zmienia tylko flagę epidemii w dniach prognozy; nie jest prognozą przebiegu epidemii. Przy zmienionym scenariuszu metryki pokazują, jak bardzo prognoza rozminęłaby się z rzeczywistością.
- **Jeden okres egzaminacyjny** ({cfg.HORIZON} dni, z krótkim okresem epidemii na początku). Różnice między modelami rzędu kilku procent nie są rozstrzygające.
- **Bias.** Zobacz kartę „Bias”: dodatni oznacza, że model zawyża sprzedaż. W tabeli wszystkich modeli widać, że w okresie egzaminacyjnym bias jest dodatni u modeli statystycznych i ML.
- **Pasmo 80%** wyliczono z błędów walidacji kroczącej na treningu (jedno pasmo o stałej szerokości). Nie uwzględnia biasu widocznego na egzaminie. Pasma nie ma dla widoków produktów i dla baseline'ów.
- **Widoki produktów** pokazują trzy najlepiej sprzedające się produkty z okresu treningowego; prognozy pochodzą z globalnego LightGBM dobranego na sumie, a nie na produktach.
- Cena i ceny konkurencji nie są używane w modelach (opis w README).
"""
    )
