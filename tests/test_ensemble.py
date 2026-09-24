"""Testy ensemble na małych przykładach liczonych ręcznie."""
import unittest

import pandas as pd

from src.ensemble import best_weights, combine, leave_one_fold_out, weight_grid, weight_table

MODELS = ["A", "B", "C"]


def toy_oof(c_offset: float = -37.0) -> pd.DataFrame:
    """Model A jest idealny, B stale zawyża o 100, C stale przesuwa o `c_offset`; dwa foldy po 3 dni.

    Przesunięcie -37 jest dobrane tak, żeby na siatce z krokiem 0,05 nie było drugiego
    kombinowanego rozwiązania o błędzie zero (przy -50 zerują go m.in. wagi 0,1 / 0,3 / 0,6).
    """
    y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0], index=pd.date_range("2024-01-01", periods=6))
    return pd.DataFrame({"fold": [1, 1, 1, 2, 2, 2], "y": y, "A": y, "B": y + 100, "C": y + c_offset})


class WeightGridTest(unittest.TestCase):
    def test_weights_sum_to_one_and_are_nonnegative(self):
        grid = weight_grid(3, 0.05)
        self.assertTrue(all(abs(sum(w) - 1) < 1e-9 and min(w) >= 0 for w in grid))

    def test_grid_size(self):
        self.assertEqual(len(weight_grid(3, 0.05)), 231)  # C(22, 2)
        self.assertEqual(len(weight_grid(2, 0.5)), 3)  # (0,1), (0.5,0.5), (1,0)


class BestWeightsTest(unittest.TestCase):
    def test_all_weight_goes_to_the_perfect_model(self):
        self.assertEqual(best_weights(toy_oof(), MODELS), {"A": 1.0, "B": 0.0, "C": 0.0})

    def test_offsetting_errors_are_cancelled(self):
        oof = toy_oof(c_offset=-50).drop(columns="A")  # B (+100) i C (-50): błąd znika przy wadze 1/3 dla B
        w = best_weights(oof, ["B", "C"], step=1 / 3)
        self.assertAlmostEqual(w["B"], 1 / 3)
        self.assertAlmostEqual(weight_table(oof, ["B", "C"], step=1 / 3).loc[0, "MAE"], 0.0)


class CombineTest(unittest.TestCase):
    def test_weighted_average(self):
        preds = pd.DataFrame({"A": [100.0, 200.0], "B": [200.0, 400.0]})
        self.assertEqual(combine(preds, {"A": 0.75, "B": 0.25}).tolist(), [125.0, 250.0])

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(AssertionError):
            combine(pd.DataFrame({"A": [1.0], "B": [1.0]}), {"A": 0.5, "B": 0.6})


class LeaveOneFoldOutTest(unittest.TestCase):
    def test_returns_row_per_fold_with_all_candidates(self):
        table = leave_one_fold_out(toy_oof(), MODELS)
        self.assertEqual(table.index.tolist(), [1, 2])
        self.assertEqual(set(table.columns), {"ważona", "zwykła średnia", *MODELS})
        self.assertAlmostEqual(table.loc[1, "A"], 0.0)
        self.assertAlmostEqual(table.loc[1, "ważona"], 0.0)  # wagi z drugiego foldu też wskazują A


if __name__ == "__main__":
    unittest.main()
