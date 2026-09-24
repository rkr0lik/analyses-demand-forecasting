"""Konfiguracja projektu: ścieżki, nazwy kolumn, daty podziału, horyzont i parametry modeli."""
from pathlib import Path

import pandas as pd

# Ścieżki
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "demand_forecasting.csv"

# Nazwy kolumn w pliku CSV
COL_DATE = "Date"
COL_STORE = "Store ID"
COL_PRODUCT = "Product ID"
COL_TARGET = "Units Sold"  # to prognozujemy
COL_CATEGORY = "Category"  # cechy statyczne serii (sklep × produkt)
COL_REGION = "Region"

# Zmienne zewnętrzne, które wolno podać modelowi (dla dni egzaminacyjnych to wariant oracle).
COL_EPIDEMIC = "Epidemic"
COL_PROMOTION = "Promotion"
COL_WEATHER = "Weather Condition"

# Tych kolumn nie wolno używać jako cech, bo nie znamy ich z wyprzedzeniem (wyciek danych).
FORBIDDEN_FEATURES = ["Demand", "Inventory Level", "Units Ordered", "Competitor Pricing"]

# Rabatu ("Discount") nie używamy. Po zsumowaniu do dnia jest prawie tym samym co promocja
# (korelacja 0,97, VIF 16) i w walidacji kroczącej nic nie dawał. Szczegóły w README.md.

# Podział danych i horyzont
HORIZON = 28
TRAIN_END = pd.Timestamp("2024-01-02")  # ostatni dzień treningu
TEST_START = pd.Timestamp("2024-01-03")  # pierwszy dzień zbioru egzaminacyjnego
TEST_END = pd.Timestamp("2024-01-30")  # ostatni dzień danych

# Walidacja krocząca (walk-forward), wyłącznie na danych treningowych
CV_FOLDS = 6  # liczba "próbnych egzaminów" po HORIZON dni na końcu treningu
CV_MIN_TRAIN_DAYS = 365  # pierwszy fold musi mieć co najmniej rok danych do nauki

# ARIMAX / SARIMAX: rzędy wybrane walidacją kroczącą na treningu (README.md)
ARIMAX_ORDER = (0, 1, 1)
SARIMAX_ORDER = (0, 1, 1)  # wariant porównawczy z sezonowością tygodniową
SARIMAX_SEASONAL_ORDER = (1, 0, 0, 7)

# Ridge: alpha i cykl roczny wybrane walidacją kroczącą na treningu (README.md)
RIDGE_ALPHA = 30
RIDGE_YEAR_WAVES = 3  # pary fal sin/cos dla cyklu rocznego, 0 wyłącza; używa tego tylko Ridge

# LightGBM: parametry wybrane walidacją kroczącą na treningu (README.md)
LGBM_PARAMS = {"num_leaves": 15, "n_estimators": 200, "min_child_samples": 20}

# Trzy modele składowe ensemble (nazwy kolumn w prognozach)
MODELS = ["ARIMAX", "LightGBM", "Ridge"]

# Ensemble: wagi wybrane siatką na prognozach out-of-fold z walidacji kroczącej (README.md)
ENSEMBLE_WEIGHTS = {"ARIMAX": 0.25, "LightGBM": 0.10, "Ridge": 0.65}

# Wyniki (CSV, żeby tabela i aplikacja nie liczyły prognoz od nowa)
OUTPUT_DIR = ROOT / "outputs"

# Baseline'y
SEASON_LENGTH = 7  # seasonal naive: powtarzamy ostatnie 7 dni treningu
MEAN_WINDOW = 28  # średnia: z ostatnich 28 dni treningu

# Losowość
RANDOM_STATE = 42
