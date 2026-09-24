"""Prognozy out-of-fold trzech modeli z walidacji kroczącej na treningu.

W każdym foldzie model uczy się tylko na danych sprzed okna testowego, więc nie widzi dni, które prognozuje.
Na tych prognozach dobieramy wagi ensemble. Zbiór egzaminacyjny nie jest tu używany.
"""
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import config as cfg
from src import arimax, ridge
from src.data import Split
from src.lgbm import build_panel, daily_total, fit_predict
from src.validation import walk_forward_folds

MODELS = cfg.MODELS


def oof_predictions(raw: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Jeden wiersz na dzień walidacji: fold, rzeczywista sprzedaż `y`, epidemia i prognozy trzech modeli."""
    y, X = split.y_train, split.X_train
    panel = build_panel(raw)
    frames = []
    for number, fold in enumerate(walk_forward_folds(y.index), start=1):
        end = fold.train[-1]
        X_test = X.loc[fold.test]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # znane i niegroźne ostrzeżenia
            arimax_pred = arimax.fit_forecast(y[:end], X[:end], X_test, cfg.ARIMAX_ORDER)
        lgbm_pred = daily_total(fit_predict(panel, end, fold.test, cfg.LGBM_PARAMS)).reindex(fold.test)
        ridge_pred = ridge.fit_forecast(y[:end], X[:end], X_test, cfg.RIDGE_ALPHA)
        frames.append(
            pd.DataFrame(
                {
                    "fold": number,
                    "y": y.loc[fold.test],
                    "epidemic": X_test["epidemic"],
                    "ARIMAX": arimax_pred,
                    "LightGBM": lgbm_pred,
                    "Ridge": ridge_pred,
                }
            )
        )
    return pd.concat(frames).rename_axis("date")


if __name__ == "__main__":
    from src.data import daily_sales, load_raw, split_train_test
    from src.features import daily_exog
    from src.metrics import compute_metrics

    raw = load_raw()
    oof = oof_predictions(raw, split_train_test(daily_sales(raw), daily_exog(raw)))

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    oof.to_csv(cfg.OUTPUT_DIR / "oof_predictions.csv")
    print(f"{len(oof)} dni w {oof['fold'].nunique()} foldach ({oof.index.min().date()} – {oof.index.max().date()})\n")

    rows = {m: compute_metrics(oof["y"], oof[m]) for m in MODELS}
    rows["zwykła średnia (podgląd)"] = compute_metrics(oof["y"], oof[MODELS].mean(axis=1))
    print(pd.DataFrame(rows).T.round(1).to_string())

    errors = oof[MODELS].sub(oof["y"], axis=0)
    print("\nkorelacja błędów modeli:")
    print(errors.corr().round(2).to_string())
