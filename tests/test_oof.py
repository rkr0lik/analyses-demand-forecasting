"""Testy prognoz out-of-fold: układ w czasie i brak sięgania w okres egzaminacyjny."""
import unittest

import config as cfg
from src.data import daily_sales, load_raw, split_train_test
from src.features import daily_exog
from src.oof import MODELS, oof_predictions


@unittest.skipUnless(cfg.DATA_PATH.exists(), "brak pliku z danymi (demand_forecasting.csv)")
class OofTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = load_raw()
        cls.split = split_train_test(daily_sales(raw), daily_exog(raw))
        cls.oof = oof_predictions(raw, cls.split)

    def test_covers_all_fold_days_once(self):
        self.assertEqual(len(self.oof), cfg.CV_FOLDS * cfg.HORIZON)
        self.assertTrue(self.oof.index.is_unique)
        self.assertEqual(self.oof["fold"].nunique(), cfg.CV_FOLDS)

    def test_stays_inside_training_period(self):
        self.assertLessEqual(self.oof.index.max(), cfg.TRAIN_END)
        self.assertTrue(self.oof.index.intersection(self.split.y_test.index).empty)

    def test_has_predictions_from_every_model_without_gaps(self):
        self.assertFalse(self.oof[["y", *MODELS]].isna().any().any())

    def test_target_matches_actual_sales(self):
        self.assertTrue((self.oof["y"] == self.split.y_train.loc[self.oof.index]).all())


if __name__ == "__main__":
    unittest.main()
