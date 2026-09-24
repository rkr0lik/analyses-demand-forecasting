"""Wykres sprzedaży i prognozy (Plotly) w stylu raportu Power BI.

Zasady wykresu: jedna oś Y, cienkie linie (2 px), poziome siatki hairline (bez pionowych), legenda widoczna
(kilka serii), kolory z walidowanej pary (niebieski i pomarańczowy z domyślnej palety Power BI), szary baseline jako
element wygaszony. Podpowiedź (hover) pokazuje wszystkie wartości dla wskazanego dnia, a te same dane są w widoku tabelarycznym.
"""
import pandas as pd
import plotly.graph_objects as go

import config as cfg
from src.app_data import BASELINE_MAIN, ViewData, epidemic_runs, exam_days_in

ACTUAL, FORECAST, BASELINE, EPIDEMIC_WASH = "#118DFF", "#E66C37", "#8A8886", "rgba(138,136,134,0.16)"
INK, GRID, AXIS = "#605E5C", "#EDEBE9", "#C8C6C4"
BAND_FILL = "rgba(230,108,55,0.16)"  # ton prognozy, ok. 16% krycia (lekka wstęga, nie blok)
SEPARATORS = ", "  # przecinek dziesiętny, twarda spacja jako separator tysięcy


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def build_figure(data: ViewData, start: pd.Timestamp, end: pd.Timestamp) -> go.Figure:
    """Rzeczywista sprzedaż, prognoza z pasmem i baseline w wybranym zakresie dat, z zacieniowaną epidemią."""
    fig = go.Figure()
    actual = data.actual[start:end]
    fig.add_trace(go.Scatter(x=actual.index, y=actual, name="Sprzedaż rzeczywista", mode="lines",
                             line=dict(color=ACTUAL, width=2), hovertemplate="%{y:,.0f}"))

    days = exam_days_in(data, start, end)
    if len(days):
        forecast = data.forecast.loc[days]
        fig.add_trace(go.Scatter(x=days, y=forecast, name=f"Prognoza: {data.model}", mode="lines",
                                 line=dict(color=FORECAST, width=2), hovertemplate="%{y:,.0f}"))
        if data.band is not None:
            band = data.band.loc[days]
            fig.add_trace(go.Scatter(x=days, y=band["upper"], name="Pasmo 80%: górna granica", mode="lines", line=dict(width=0),
                                     legendgroup="band", showlegend=False, hovertemplate="%{y:,.0f}"))
            fig.add_trace(go.Scatter(x=days, y=band["lower"], name="Pasmo 80%: dolna granica", mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor=BAND_FILL, legendgroup="band", showlegend=False,
                                     hovertemplate="%{y:,.0f}"))
            fig.add_trace(go.Scatter(x=[None], y=[None], name="Pasmo niepewności 80%", mode="markers", legendgroup="band",
                                     marker=dict(symbol="square", size=10, color=BAND_FILL), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=days, y=data.baselines[BASELINE_MAIN].loc[days], name=f"Baseline: {BASELINE_MAIN}", mode="lines",
                                 line=dict(color=BASELINE, width=2), hovertemplate="%{y:,.0f}"))

    # Epidemia: zacieniowane tło (bez własnej serii); wpis w legendzie przez pusty znacznik.
    runs = [(max(s, start), min(e, end)) for s, e in epidemic_runs(data.epidemic) if s <= end and e >= start]
    for s, e in runs:
        fig.add_vrect(x0=_iso(s - pd.Timedelta(hours=12)), x1=_iso(e + pd.Timedelta(hours=12)), fillcolor=EPIDEMIC_WASH,
                      line_width=0, layer="below")
    if runs:
        fig.add_trace(go.Scatter(x=[None], y=[None], name="Dni z epidemią", mode="markers",
                                 marker=dict(symbol="square", size=10, color=EPIDEMIC_WASH), hoverinfo="skip"))

    # Granica między historią a prognozą.
    if start <= cfg.TEST_START <= end:
        x = _iso(cfg.TEST_START - pd.Timedelta(hours=12))
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=AXIS, width=1))
        fig.add_annotation(x=x, y=1, yref="paper", text="Początek prognozy", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11, color=INK))

    fig.update_layout(
        height=420, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="white", plot_bgcolor="white", separators=SEPARATORS,
        font=dict(family='"Segoe UI", system-ui, sans-serif', size=12, color=INK), hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.14, yanchor="bottom", font=dict(size=12, color=INK)),
        hoverlabel=dict(font=dict(size=12)),
    )
    fig.update_xaxes(range=[_iso(start - pd.Timedelta(hours=12)), _iso(end + pd.Timedelta(hours=12))], showgrid=False,
                     linecolor=AXIS, tickformat="%d.%m.%Y", showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor="#8A8886", spikedash="solid", spikesnap="cursor")
    fig.update_yaxes(title=dict(text="Sprzedaż dzienna, szt.", font=dict(size=12)), tickformat=",d", gridcolor=GRID, gridwidth=1,
                     zeroline=False, linecolor="rgba(0,0,0,0)", rangemode="tozero")
    return fig
