"""Uruchamia cały pipeline od zera i zapisuje wszystkie wyniki do outputs/ (potrzebny plik demand_forecasting.csv).

Każdy krok to osobny skrypt z src/ (`python -m src.<nazwa>`), uruchamiany w kolejności zależności. Parametry modeli
(rzędy ARIMAX, alpha Ridge, parametry LightGBM, wagi ensemble) są już w config.py, więc czasochłonne przeszukiwanie
siatek (src.arimax, src.ridge, src.lgbm, src.ensemble) nie jest częścią tego skryptu.

Użycie: python -m src.run_all
"""
import subprocess
import sys
import time

import config as cfg

# (moduł, co robi); kolejność wynika z zależności plików w outputs/
STEPS = [
    ("src.export_actuals", "zagregowana rzeczywista sprzedaż (suma dzienna, per produkt, flaga epidemii)"),
    ("src.run_models", "prognozy egzaminacyjne: ARIMAX, SARIMAX, Ridge, LightGBM i ensemble"),
    ("src.oof", "prognozy out-of-fold z walidacji kroczącej (do wag i pasm)"),
    ("src.compare", "tabela porównawcza wszystkich modeli"),
    ("src.bonus", "prognozy i metryki dla 3 najlepszych produktów"),
    ("src.intervals", "pasma niepewności 80%"),
    ("src.scenarios", "scenariusze długości epidemii"),
    ("src.audit", "kontrola poprawności: przeliczenie metryk, efekt miesiąca, istotność różnic"),
    ("src.docs_tables", "tabele i podsumowania w README.md, NAUKA.md i DOKUMENTACJA.md"),
]


def main() -> None:
    started = time.time()
    for module, description in STEPS:
        step_started = time.time()
        print(f"{module:<20} {description} ...", end=" ", flush=True)
        result = subprocess.run([sys.executable, "-W", "ignore", "-m", module], cwd=cfg.ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print("BŁĄD\n")
            print(result.stdout + result.stderr)
            sys.exit(result.returncode)
        print(f"ok ({time.time() - step_started:.0f} s)", flush=True)
    print(f"\nGotowe w {time.time() - started:.0f} s. Wyniki w {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
