"""Zapis zagregowanej rzeczywistej sprzedaży do outputs/, żeby wykresy w aplikacji działały bez pliku z danymi.

Zapisujemy wyłącznie agregaty: sumę dzienną i sumę dzienną per produkt (po sklepach) oraz flagę epidemii
(0/1 na dzień; potrzebna do metryk osobno dla epidemii i do zacieniowania jej na wykresie).
Surowe wiersze (sklep × produkt × dzień) i pozostałe kolumny zostają w pliku z danymi, którego nie ma w repozytorium.
"""
import pandas as pd

import config as cfg
from src.bonus import product_daily_sales
from src.data import daily_sales, load_raw
from src.features import daily_exog


def export_actuals(raw: pd.DataFrame, output_dir=cfg.OUTPUT_DIR) -> None:
    """Zapisz daily_sales.csv (date, units_sold, epidemic) i product_daily_sales.csv (date + kolumna na produkt)."""
    output_dir.mkdir(exist_ok=True)
    daily_sales(raw).to_frame().assign(epidemic=daily_exog(raw)["epidemic"]).to_csv(
        output_dir / "daily_sales.csv", index_label="date"
    )
    product_daily_sales(raw).to_csv(output_dir / "product_daily_sales.csv", index_label="date")


if __name__ == "__main__":
    export_actuals(load_raw())
    print(f"zapisano: {cfg.OUTPUT_DIR / 'daily_sales.csv'}, {cfg.OUTPUT_DIR / 'product_daily_sales.csv'}")
