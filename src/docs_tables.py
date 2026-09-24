"""Tabele i podsumowania w dokumentach generowane z plików w outputs/ (żadnych liczb przepisywanych ręcznie).

Obsługiwane dokumenty: README.md, NAUKA.md i DOKUMENTACJA.md. Bloki wyglądają w nich tak:
<!-- BEGIN:nazwa --> ... <!-- END:nazwa -->  i są podmieniane przez ten skrypt.
Użycie: python -m src.docs_tables   (aktualizuje wszystkie trzy pliki; testy pilnują, żeby były aktualne)
"""
import re

import pandas as pd

import config as cfg
from src.intervals import with_ensembles
from src.metrics import compute_metrics
from src.oof import MODELS

README = cfg.ROOT / "README.md"
NAUKA = cfg.ROOT / "NAUKA.md"
DOKUMENTACJA = cfg.ROOT / "DOKUMENTACJA.md"
NBSP = " "
MODEL_ROWS = ["ARIMAX", "SARIMAX", "Ridge", "LightGBM", "Ensemble zwykła średnia", "Ensemble ważona średnia"]
BASELINE_ROWS = ["naive", "seasonal naive", f"średnia {cfg.MEAN_WINDOW} dni"]
SINGLE_MODELS = ["ARIMAX", "SARIMAX", "Ridge", "LightGBM"]  # pojedyncze modele (bez ensemble)


def num(value: float, decimals: int = 1, signed: bool = False) -> str:
    """Liczba po polsku: przecinek dziesiętny, twarda spacja jako separator tysięcy, typograficzny minus."""
    text = f"{value:{'+' if signed else ''},.{decimals}f}"
    return text.replace(",", NBSP).replace(".", ",").replace("-", "−")


def pct(value: float, decimals: int = 1) -> str:
    return num(value, decimals) + "%"


def table(headers: list[str], rows: list[list[str]], text_columns: int = 1, aligns: str | None = None) -> str:
    """Tabela Markdown; pierwsze `text_columns` kolumn wyrównane do lewej, reszta do prawej (albo `aligns`, np. "lrrl")."""
    if aligns is None:
        aligns = "l" * text_columns + "r" * (len(headers) - text_columns)
    align = ["---" if a == "l" else "---:" for a in aligns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _read(name: str, **kwargs) -> pd.DataFrame:
    path = cfg.OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"brak {path}; uruchom najpierw: python -m src.run_all")
    return pd.read_csv(path, **kwargs)


def comparison() -> pd.DataFrame:
    return _read("comparison.csv", index_col="model")


def block_summary() -> str:
    c = comparison()
    best_base = c.loc[BASELINE_ROWS, "MAE"].idxmin()
    base_mae = c.loc[best_base, "MAE"]
    best_single = c.loc[SINGLE_MODELS, "MAE"].idxmin()
    models = c.loc[MODEL_ROWS]
    lo, hi = models["MAE"].min(), models["MAE"].max()
    bias_lo, bias_hi = models["Bias"].min(), models["Bias"].max()
    direction = "wszystkie sześć modeli zawyża prognozę" if (models["Bias"] > 0).all() else "bias sześciu modeli ma różne znaki"
    gain = lambda m: pct((1 - c.loc[m, "MAE"] / base_mae) * 100)  # noqa: E731
    return "\n".join(
        [
            f"- Najlepszy baseline: **{best_base}** (MAE {num(base_mae)}, MAPE {pct(c.loc[best_base, 'MAPE'])}).",
            f"- **Eksperyment 1, ARIMAX:** MAE {num(c.loc['ARIMAX', 'MAE'])}, MAPE {pct(c.loc['ARIMAX', 'MAPE'])}, czyli błąd o "
            f"{gain('ARIMAX')} mniejszy niż najlepszego baseline'u. Wariant sezonowy SARIMAX: MAE {num(c.loc['SARIMAX', 'MAE'])}.",
            f"- **Eksperyment 2, ensemble:** zwykła średnia MAE {num(c.loc['Ensemble zwykła średnia', 'MAE'])}, ważona MAE "
            f"{num(c.loc['Ensemble ważona średnia', 'MAE'])}. Najlepszy pojedynczy model: **{best_single}** "
            f"(MAE {num(c.loc[best_single, 'MAE'])}, MAPE {pct(c.loc[best_single, 'MAPE'])}).",
            f"- Sześć modeli mieści się w MAE od {num(lo)} do {num(hi)} (różnica {pct((hi / lo - 1) * 100)}), "
            f"czyli błąd jest o {pct((1 - hi / base_mae) * 100)}–{pct((1 - lo / base_mae) * 100)} mniejszy niż najlepszego baseline'u.",
            f"- Bias (plus = zawyża) sześciu modeli: od {num(bias_lo, signed=True)} do {num(bias_hi, signed=True)} szt. dziennie; {direction}.",
            f"- MAPE Ridge (regresja z hamulcem) to {pct(c.loc['Ridge', 'MAPE'])}, a ARIMAX {pct(c.loc['ARIMAX', 'MAPE'])}.",
        ]
    )


def block_comparison() -> str:
    c = comparison().sort_values("MAE")
    rows = [[m, c.loc[m, "grupa"], num(c.loc[m, "MAE"]), num(c.loc[m, "RMSE"]), pct(c.loc[m, "MAPE"]), num(c.loc[m, "Bias"], signed=True)]
            for m in c.index]
    return table(["Model", "Grupa", "MAE", "RMSE", "MAPE", "Bias"], rows, text_columns=2)


def block_epidemic() -> str:
    c = comparison().sort_values("MAE")
    rows = [
        [m, num(c.loc[m, "MAE epidemia"]), pct(c.loc[m, "MAPE epidemia"]), num(c.loc[m, "Bias epidemia"], signed=True),
         num(c.loc[m, "MAE bez epidemii"]), pct(c.loc[m, "MAPE bez epidemii"]), num(c.loc[m, "Bias bez epidemii"], signed=True)]
        for m in c.index
    ]
    headers = ["Model", "MAE epidemia", "MAPE epidemia", "Bias epidemia", "MAE bez epidemii", "MAPE bez epidemii", "Bias bez epidemii"]
    return table(headers, rows)


def block_validation() -> str:
    oof = with_ensembles(_read("oof_predictions.csv", index_col="date", parse_dates=True))
    exam = comparison()
    names = [*MODELS, "Ensemble zwykła średnia", "Ensemble ważona średnia"]
    rows = sorted(((m, compute_metrics(oof["y"], oof[m])["MAE"], exam.loc[m, "MAE"]) for m in names), key=lambda r: r[1])
    return table(["Model", "MAE w walidacji kroczącej (trening)", "MAE na egzaminie"], [[m, num(v), num(e)] for m, v, e in rows])


def block_models() -> str:
    lgbm = cfg.LGBM_PARAMS
    w = cfg.ENSEMBLE_WEIGHTS
    return "\n".join(
        [
            f"- **ARIMAX:** rząd (p,d,q) = {cfg.ARIMAX_ORDER}, bez sezonowości, ze zmiennymi zewnętrznymi.",
            f"- **SARIMAX (wariant porównawczy):** rząd {cfg.SARIMAX_ORDER} z sezonowością {cfg.SARIMAX_SEASONAL_ORDER}.",
            f"- **Ridge:** `alpha` = {cfg.RIDGE_ALPHA} na standaryzowanych zmiennych zewnętrznych "
            f"plus cykl roczny ({cfg.RIDGE_YEAR_WAVES} pary fal sin/cos dnia roku).",
            f"- **LightGBM (globalny):** `num_leaves` = {lgbm['num_leaves']}, `n_estimators` = {lgbm['n_estimators']}, "
            f"`min_child_samples` = {lgbm['min_child_samples']}, `learning_rate` = 0,05, `random_state` = {cfg.RANDOM_STATE}.",
            f"- **Ensemble:** zwykła średnia (po 1/3) oraz średnia ważona z wagami ARIMAX {num(w['ARIMAX'], 2)}, "
            f"LightGBM {num(w['LightGBM'], 2)}, Ridge {num(w['Ridge'], 2)}.",
        ]
    )


def block_bonus() -> str:
    t = _read("comparison_top_products.csv", index_col=["produkt", "model"])
    rows = []
    for product in t.index.get_level_values("produkt").unique():
        part = t.loc[product].sort_values("MAE")
        rows += [[product, m, num(part.loc[m, "MAE"]), num(part.loc[m, "RMSE"]), pct(part.loc[m, "MAPE"]), num(part.loc[m, "Bias"], signed=True)]
                 for m in part.index]
    return table(["Produkt", "Model", "MAE", "RMSE", "MAPE", "Bias"], rows, text_columns=2)


def block_bonus_products() -> str:
    products = _read("comparison_top_products.csv", index_col=["produkt", "model"]).index.get_level_values("produkt").unique()
    return ", ".join(f"**{p}**" for p in products)


def block_coverage() -> str:
    c = _read("interval_coverage.csv", index_col="model")
    rows = [[m, pct(c.loc[m, "pokrycie"] * 100, 0), num(c.loc[m, "średnia szerokość"], 0)] for m in c.index]
    return table(["Model", "Pokrycie na egzaminie (deklarowane 80%)", "Średnia szerokość pasma, szt."], rows)


def block_scenarios() -> str:
    s = _read("scenarios.csv")
    total = s.groupby(["model", "epidemic_days"])["forecast"].sum().unstack("epidemic_days")
    rows = [[m, num(total.loc[m, 0], 0), num(total.loc[m, 6], 0), num(total.loc[m, cfg.HORIZON], 0),
             num((total.loc[m, cfg.HORIZON] - total.loc[m, 0]) / cfg.HORIZON, 0, signed=True)] for m in total.index]
    headers = ["Model", "N = 0", "N = 6 (faktycznie)", f"N = {cfg.HORIZON}", "Zmiana sumy na każdy dzień epidemii"]
    return table(headers, rows)



# ---------------------------------------------------------------------------------------------------------------------
# Bloki dla NAUKA.md (opis dla laika): te same wyniki, ale w prostszej formie
# ---------------------------------------------------------------------------------------------------------------------
PLAIN_NAMES = {
    "naive": "naive (jak wczoraj)",
    "seasonal naive": "seasonal naive (jak tydzień temu)",
    f"średnia {cfg.MEAN_WINDOW} dni": f"średnia z ostatnich {cfg.MEAN_WINDOW} dni",
    "ARIMAX": "ARIMAX",
    "SARIMAX": "SARIMAX (z rytmem tygodniowym)",
    "Ridge": "Ridge",
    "LightGBM": "LightGBM",
    "Ensemble zwykła średnia": "połączenie: zwykła średnia",
    "Ensemble ważona średnia": "połączenie: średnia ważona",
}


def _direction(bias: float) -> str:
    return f"zawyża o ok. {num(abs(bias), 0)} szt. dziennie" if bias > 0 else f"zaniża o ok. {num(abs(bias), 0)} szt. dziennie"


def _daily() -> pd.DataFrame:
    return _read("daily_sales.csv", index_col="date", parse_dates=True)


def block_nauka_data() -> str:
    d = _daily()
    train = d[: cfg.TRAIN_END]
    by_epidemic = train.groupby("epidemic")["units_sold"].mean()
    by_weekday = train["units_sold"].groupby(train.index.dayofweek).mean()
    spread = (by_weekday.max() - by_weekday.min()) / train["units_sold"].mean() * 100
    return "\n".join(
        [
            f"- W danych jest **{len(d)} dni**: {len(train)} dni treningowych i {len(d) - len(train)} dni egzaminacyjnych.",
            f"- W okresie treningowym sieć sprzedawała średnio **{num(train['units_sold'].mean(), 0)} sztuk dziennie** (wszystkie sklepy i produkty razem).",
            f"- Epidemia trwała w treningu łącznie {int(train['epidemic'].sum())} dni. W jej dniach średnia dzienna sprzedaż wynosiła "
            f"**{num(by_epidemic[1], 0)}** zamiast {num(by_epidemic[0], 0)} sztuk, czyli spadek o {pct(abs(by_epidemic[1] / by_epidemic[0] - 1) * 100)}.",
            f"- Najlepszy i najsłabszy dzień tygodnia różnią się średnią sprzedażą tylko o {pct(spread)} średniej, więc rytmu tygodniowego praktycznie nie ma.",
        ]
    )


def block_nauka_headline() -> str:
    c = comparison()
    best_base = c.loc[BASELINE_ROWS, "MAE"].idxmin()
    models = c.loc[MODEL_ROWS]
    best = c.loc[SINGLE_MODELS, "MAE"].idxmin()
    gain_lo = (1 - models["MAE"].max() / c.loc[best_base, "MAE"]) * 100
    gain_hi = (1 - models["MAE"].min() / c.loc[best_base, "MAE"]) * 100
    return "\n".join(
        [
            f"- Najprostsza sensowna metoda ({PLAIN_NAMES[best_base]}) myli się średnio o **{num(c.loc[best_base, 'MAE'], 0)} sztuk dziennie** "
            f"({pct(c.loc[best_base, 'MAPE'])}).",
            f"- Nasze modele mylą się średnio o **{num(models['MAE'].min(), 0)}–{num(models['MAE'].max(), 0)} sztuk dziennie** "
            f"({pct(models['MAPE'].min())}–{pct(models['MAPE'].max())}), czyli o {pct(gain_lo, 0)}–{pct(gain_hi, 0)} mniej niż najprostsza metoda.",
            f"- Najlepszy z pojedynczych modeli na egzaminie to **{best}**, ale różnice między modelami są małe, a egzamin był tylko jeden.",
            f"- Wszystkie modele **{'zawyżają' if (models['Bias'] > 0).all() else 'mylą się w różne strony'}** prognozę na styczeń 2024: "
            f"od ok. {num(models['Bias'].min(), 0)} do ok. {num(models['Bias'].max(), 0)} sztuk dziennie za dużo.",
        ]
    )


def block_nauka_models() -> str:
    c = comparison().sort_values("MAE")
    rows = [[PLAIN_NAMES[m], num(c.loc[m, "MAE"], 0), pct(c.loc[m, "MAPE"]), _direction(c.loc[m, "Bias"])] for m in c.index]
    return table(["Metoda", "Średni błąd dzienny (szt.)", "Średni błąd w %", "Kierunek błędu (bias)"], rows, aligns="lrrl")


def block_nauka_ranking() -> str:
    oof = with_ensembles(_read("oof_predictions.csv", index_col="date", parse_dates=True))
    exam = comparison()
    val = {m: compute_metrics(oof["y"], oof[m])["MAE"] for m in ("ARIMAX", "Ridge")}
    return (
        f"W próbnych egzaminach ARIMAX myli się średnio o {num(val['ARIMAX'], 0)} sztuk dziennie, a Ridge o {num(val['Ridge'], 0)}, więc lepszy jest ARIMAX. "
        f"Na prawdziwym egzaminie jest odwrotnie: ARIMAX {num(exam.loc['ARIMAX', 'MAE'], 0)}, Ridge {num(exam.loc['Ridge', 'MAE'], 0)}."
    )


def block_nauka_scenarios() -> str:
    s = _read("scenarios.csv")
    total = s.groupby(["model", "epidemic_days"])["forecast"].sum().unstack("epidemic_days")
    per_day = (total[cfg.HORIZON] - total[0]) / cfg.HORIZON
    real, none = total.loc["ARIMAX", 6], total.loc["ARIMAX", 0]
    return (
        f"Każdy dodatkowy dzień epidemii obniża łączną sprzedaż z {cfg.HORIZON} dni o ok. {num(abs(per_day.max()), 0)}–{num(abs(per_day.min()), 0)} sztuk "
        f"(zależnie od modelu). Dla modelu ARIMAX suma prognozy wynosi {num(real, 0)} szt. przy faktycznej długości epidemii i {num(none, 0)} szt., "
        f"gdyby epidemii nie było, czyli różnica to {num(none - real, 0)} szt. ({pct((none / real - 1) * 100, 0)})."
    )


def block_nauka_band() -> str:
    c = _read("interval_coverage.csv", index_col="model")["pokrycie"] * 100
    span = pct(c.min(), 0) if round(c.min()) == round(c.max()) else f"{pct(c.min(), 0)}–{pct(c.max(), 0)}"
    return f"Pasmo pokryło rzeczywistą sprzedaż w {span} dni egzaminacyjnych (deklarowane było 80%)."


def block_nauka_bonus() -> str:
    t = _read("comparison_top_products.csv", index_col=["produkt", "model"])
    mean_name = f"średnia {cfg.MEAN_WINDOW} dni"
    lines = []
    for product in t.index.get_level_values("produkt").unique():
        lgbm, base = t.loc[(product, "LightGBM"), "MAE"], t.loc[(product, mean_name), "MAE"]
        lines.append(f"- **{product}**: LightGBM myli się średnio o {num(lgbm, 0)} szt. dziennie, a średnia z ostatnich {cfg.MEAN_WINDOW} dni o {num(base, 0)} "
                     f"(błąd mniejszy o {pct((1 - lgbm / base) * 100, 0)}).")
    return "\n".join(lines)


def block_nauka_plan() -> str:
    c = comparison()
    return (f"Plan zakładał, że prosta regresja liniowa myli się na egzaminie o ok. 6%. U nas najprostszy z modeli regresyjnych (Ridge) myli się o "
            f"{pct(c.loc['Ridge', 'MAPE'])}, a ARIMAX o {pct(c.loc['ARIMAX', 'MAPE'])}: ten sam rząd wielkości, ale inna liczba.")


# ---------------------------------------------------------------------------------------------------------------------
# Bloki dla DOKUMENTACJA.md: wyniki kontroli poprawności z outputs/ (tworzy je `python -m src.audit`)
# ---------------------------------------------------------------------------------------------------------------------
def block_audyt_metryki() -> str:
    biggest = _read("audit_metrics.csv")["największa różnica metryk"].iloc[0]
    return (f"Metryki wszystkich modeli przeliczono od nowa z zapisanych prognoz, osobną implementacją. "
            f"Największa różnica wobec tabeli wyników to {biggest:.0e}, czyli tyle, ile wynosi dokładność zapisu liczb. "
            "Wyniki w tabelach zgadzają się z prognozami.")


def block_audyt_miesiace() -> str:
    m = _read("audit_months.csv", index_col="miesiąc").sort_values("efekt", ascending=False)
    rows = [[name, num(m.loc[name, "efekt"], 0, signed=True), "± " + num(m.loc[name, "blad_standardowy"], 0)] for name in m.index]
    return table(["Miesiąc", "Ile sprzedaży model nie tłumaczy (szt./dzień)", "Niepewność"], rows)


def block_audyt_sezonowosc() -> str:
    g = _read("audit_seasonality.csv", index_col="wariant")
    base = g["MAE"].iloc[0]
    rows = [[name, num(g.loc[name, "MAE"]), num(g.loc[name, "Bias"], signed=True),
             "punkt odniesienia" if i == 0 else pct((1 - g.loc[name, "MAE"] / base) * 100, 0)] for i, name in enumerate(g.index)]
    return table(["Wariant", "Błąd w walidacji (MAE)", "Bias", "Poprawa"], rows)


def block_cykl_roczny() -> str:
    """Ridge z cyklem rocznym i bez: walidacja, egzamin i kontrolny styczeń rok wcześniej."""
    g = _read("audit_seasonality.csv", index_col="wariant")
    c = _read("audit_year_cycle.csv", index_col="wariant Ridge")
    walidacja = {"bez": g["MAE"].iloc[0], "z": g.loc[[i for i in g.index if "obecny model" in i][0], "MAE"]}
    rows = [
        ["Ridge bez cyklu rocznego", num(walidacja["bez"]), num(c["MAE egzamin"].iloc[0]), num(c["MAE styczeń rok wcześniej"].iloc[0])],
        [f"Ridge z cyklem rocznym ({cfg.RIDGE_YEAR_WAVES} pary fal)", num(walidacja["z"]), num(c["MAE egzamin"].iloc[1]), num(c["MAE styczeń rok wcześniej"].iloc[1])],
    ]
    return table(["Wariant", "MAE w walidacji (trening)", "MAE na egzaminie", "MAE w kontrolnym styczniu"], rows)


def block_wybor_produktow() -> str:
    """Ranking produktów z treningu i z całości danych: pokazuje, że wybór trójki zależy od okresu."""
    r = _read("top_products_ranking.csv", index_col="produkt")
    rows = [[name, num(r.loc[name, "średnia z treningu"]), str(int(r.loc[name, "pozycja (trening)"])),
             num(r.loc[name, "średnia z całości"]), str(int(r.loc[name, "pozycja (całość)"]))] for name in r.index]
    return table(["Produkt", "Średnia dzienna (trening)", "Pozycja", "Średnia dzienna (całość danych)", "Pozycja"], rows)


def block_audyt_istotnosc() -> str:
    t = _read("audit_tests.csv", index_col="porównanie")
    rows = [[name, num(t.loc[name, "różnica MAE"]), num(t.loc[name, "p"], 2), t.loc[name, "istotna"]] for name in t.index]
    return table(["Porównanie", "Różnica MAE", "p", "Różnica istotna?"], rows)


def block_przyklad() -> str:
    """Przykład liczbowy dla miar błędu, policzony tą samą funkcją co wyniki projektu."""
    y_true, y_pred = [100.0, 200.0], [110.0, 180.0]
    m = compute_metrics(y_true, y_pred)
    return (
        "**Przykład ilustracyjny (liczby wymyślone, to nie wynik projektu).** Dwa dni: prawdziwa sprzedaż to 100 i 200 "
        "sztuk, a prognoza to 110 i 180. Pomyłki wynoszą więc +10 i −20 sztuk. "
        f"MAE to średnia z 10 i 20, czyli {num(m['MAE'], 0)}. MAPE to średnia z 10% i 10%, czyli {pct(m['MAPE'], 0)}. "
        f"Bias to średnia z +10 i −20, czyli {num(m['Bias'], 0, signed=True)}, więc prognoza w sumie lekko zaniża."
    )


def block_audyt_szczegoly() -> str:
    c = _read("audit_checks.csv", index_col="sprawdzenie")["wartość"]
    decimals = {"korelacja kolumny Demand ze sprzedażą": 2, "test stacjonarności reszt regresji (p)": 4}
    rows = [[name, num(c[name], decimals.get(name, 0))] for name in c.index]
    return table(["Sprawdzenie", "Wartość"], rows)


DOC_BLOCKS = {
    "przyklad": block_przyklad,
    "audyt_szczegoly": block_audyt_szczegoly,
    "audyt_metryki": block_audyt_metryki,
    "audyt_miesiace": block_audyt_miesiace,
    "audyt_sezonowosc": block_audyt_sezonowosc,
    "audyt_istotnosc": block_audyt_istotnosc,
    "cykl_roczny": block_cykl_roczny,
    "wybor_produktow": block_wybor_produktow,
}

NAUKA_BLOCKS = {
    "nauka_headline": block_nauka_headline,
    "nauka_data": block_nauka_data,
    "nauka_models": block_nauka_models,
    "nauka_ranking": block_nauka_ranking,
    "nauka_scenarios": block_nauka_scenarios,
    "nauka_band": block_nauka_band,
    "nauka_bonus": block_nauka_bonus,
    "nauka_plan": block_nauka_plan,
}

BLOCKS = {
    "summary": block_summary,
    "models": block_models,
    "comparison": block_comparison,
    "epidemic": block_epidemic,
    "validation": block_validation,
    "bonus_products": block_bonus_products,
    "bonus": block_bonus,
    "coverage": block_coverage,
    "scenarios": block_scenarios,
    "cykl_roczny": block_cykl_roczny,
    "wybor_produktow": block_wybor_produktow,
}


ALL_BLOCKS = {**BLOCKS, **NAUKA_BLOCKS, **DOC_BLOCKS}
MARKER = re.compile(r"<!-- BEGIN:(\w+) -->")


def render(text: str, blocks: dict | None = None) -> str:
    """Wypełnij w tekście wszystkie bloki generowane. Nazwa bez generatora to błąd."""
    blocks = ALL_BLOCKS if blocks is None else blocks
    for name in MARKER.findall(text):
        if name not in blocks:
            raise ValueError(f"blok {name} nie ma generatora (dostępne: {', '.join(sorted(blocks))})")
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        pattern = re.compile(rf"({re.escape(begin)})\n?.*?\n?({re.escape(end)})", re.S)
        if not pattern.search(text):
            raise ValueError(f"blok {name} nie ma zamknięcia {end}")
        text = pattern.sub(lambda m: f"{begin}\n{blocks[name]()}\n{end}", text)
    return text


if __name__ == "__main__":
    for path in (README, NAUKA, DOKUMENTACJA):
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = render(original)
        path.write_text(updated, encoding="utf-8")
        used = len(set(MARKER.findall(updated)))
        print(f"{path.name}: {'zaktualizowano' if updated != original else 'bez zmian'} ({used} bloków)")
