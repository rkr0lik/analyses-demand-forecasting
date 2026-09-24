"""Zapisuje zagregowaną sprzedaż do outputs/, żeby aplikacja działała bez pliku z danymi.

Do plików trafia suma dzienna, suma dzienna dla każdego produktu (po sklepach) i flaga epidemii 0/1.
Flaga jest potrzebna do metryk w podziale na epidemię i do zacieniowania wykresu. Surowe wiersze
(sklep × produkt × dzień) zostają w CSV, którego nie ma w repozytorium.
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
