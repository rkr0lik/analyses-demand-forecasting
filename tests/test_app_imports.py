"""Aplikacja nie może importować bibliotek do uczenia modeli (ciężki obraz Dockera, zbędne zależności)."""
import subprocess
import sys
import unittest

import config as cfg

HEAVY = ["statsmodels", "sklearn", "lightgbm", "scipy", "matplotlib"]


class AppImportsTest(unittest.TestCase):
    def test_app_modules_do_not_import_the_ml_stack(self):
        code = (
            "import sys, app.charts, src.app_data\n"
            f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
        )
        result = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=cfg.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "", f"aplikacja importuje: {result.stdout.strip()}")


if __name__ == "__main__":
    unittest.main()
