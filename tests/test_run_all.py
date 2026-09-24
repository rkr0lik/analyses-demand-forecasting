"""Testy kolejności kroków w run_all: moduły istnieją, a zależności między plikami są zachowane."""
import importlib.util
import unittest

from src.run_all import STEPS

ORDER = [module for module, _ in STEPS]


class RunAllTest(unittest.TestCase):
    def test_every_step_is_an_existing_runnable_module(self):
        for module in ORDER:
            spec = importlib.util.find_spec(module)
            self.assertIsNotNone(spec, module)
            self.assertIn('__name__ == "__main__"', open(spec.origin, encoding="utf-8").read(), f"{module} nie ma bloku uruchomieniowego")

    def test_no_step_is_repeated(self):
        self.assertEqual(len(ORDER), len(set(ORDER)))

    def test_dependencies_come_first(self):
        after = {
            "src.compare": ["src.run_models"],
            "src.bonus": ["src.run_models", "src.export_actuals"],
            "src.intervals": ["src.oof", "src.compare"],
            "src.scenarios": ["src.run_models"],
            "src.audit": ["src.compare", "src.run_models"],
            "src.docs_tables": ["src.compare", "src.bonus", "src.intervals", "src.scenarios", "src.oof", "src.audit"],
        }
        for step, prerequisites in after.items():
            for prerequisite in prerequisites:
                self.assertLess(ORDER.index(prerequisite), ORDER.index(step), f"{prerequisite} musi być przed {step}")


if __name__ == "__main__":
    unittest.main()
