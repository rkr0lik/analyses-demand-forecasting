"""Testy walidacji kroczącej: układ okien i brak sięgania w okres egzaminacyjny."""
import unittest
from unittest import mock

import pandas as pd

import config as cfg
from src.data import daily_sales, load_raw, split_train_test
from src.features import daily_exog
from src.validation import walk_forward_folds


class WalkForwardLayoutTest(unittest.TestCase):
    def setUp(self):
        # Krótka oś kończąca się w TRAIN_END; minimum uczenia obniżone, żeby przykład był mały.
        self.index = pd.date_range(end=cfg.TRAIN_END, periods=100)
        patcher = mock.patch.object(cfg, "CV_MIN_TRAIN_DAYS", 30)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folds = walk_forward_folds(self.index, horizon=10, n_folds=3)

    def test_number_and_length_of_test_windows(self):
        self.assertEqual(len(self.folds), 3)
        self.assertTrue(all(len(f.test) == 10 for f in self.folds))

    def test_last_test_window_ends_at_last_day(self):
        self.assertEqual(self.folds[-1].test[-1], self.index[-1])

    def test_test_windows_are_consecutive_and_do_not_overlap(self):
        for a, b in zip(self.folds, self.folds[1:]):
            self.assertEqual(b.test[0] - a.test[-1], pd.Timedelta(days=1))

    def test_train_ends_right_before_test_and_expands(self):
        for f in self.folds:
            self.assertEqual(f.test[0] - f.train[-1], pd.Timedelta(days=1))
        self.assertEqual([len(f.train) for f in self.folds], [70, 80, 90])

    def test_too_many_folds_is_rejected(self):
        with self.assertRaises(AssertionError):
            walk_forward_folds(self.index, horizon=10, n_folds=8)  # zostałoby 20 dni uczenia < 30

    def test_index_beyond_train_end_is_rejected(self):
        with self.assertRaises(AssertionError):
            walk_forward_folds(pd.date_range(end=cfg.TEST_END, periods=800), horizon=10, n_folds=3)


@unittest.skipUnless(cfg.DATA_PATH.exists(), "brak pliku z danymi (demand_forecasting.csv)")
class WalkForwardRealDataTest(unittest.TestCase):
    def test_folds_stay_inside_training_period(self):
        raw = load_raw()
        split = split_train_test(daily_sales(raw), daily_exog(raw))
        folds = walk_forward_folds(split.y_train.index)

        self.assertEqual(len(folds), cfg.CV_FOLDS)
        self.assertEqual(folds[-1].test[-1], cfg.TRAIN_END)
        for f in folds:
            self.assertLessEqual(f.test[-1], cfg.TRAIN_END)
            self.assertLess(f.test[-1], cfg.TEST_START)
            self.assertEqual(len(f.test), cfg.HORIZON)
        self.assertGreaterEqual(len(folds[0].train), cfg.CV_MIN_TRAIN_DAYS)


if __name__ == "__main__":
    unittest.main()
