"""Testy statyczne plików kontenera: co trafia do obrazu, kto go uruchamia, jak sprawdzane jest zdrowie."""
import re
import unittest

import config as cfg

DOCKERFILE = (cfg.ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (cfg.ROOT / ".dockerignore").read_text(encoding="utf-8")
ALLOWED = {"requirements-app.txt", "config.py", "src", "app", ".streamlit", "outputs"}


def dockerignore_rules() -> list[str]:
    return [line.strip() for line in DOCKERIGNORE.splitlines() if line.strip() and not line.strip().startswith("#")]


def copy_sources() -> list[str]:
    """Źródła z poleceń COPY (bez flag i celu)."""
    sources = []
    for line in DOCKERFILE.splitlines():
        if line.startswith("COPY "):
            parts = [p for p in line.split()[1:] if not p.startswith("--")]
            sources += [p.rstrip("/") for p in parts[:-1]]
    return sources


class DockerIgnoreTest(unittest.TestCase):
    def test_everything_is_excluded_by_default(self):
        self.assertEqual(dockerignore_rules()[0], "*")

    def test_only_expected_paths_are_whitelisted(self):
        whitelisted = {rule[1:].rstrip("/") for rule in dockerignore_rules() if rule.startswith("!")}
        self.assertEqual(whitelisted, ALLOWED)

    def test_whitelisted_paths_exist(self):
        for name in ALLOWED:
            self.assertTrue((cfg.ROOT / name).exists(), name)

    def test_data_file_can_not_get_into_the_image(self):
        rules = " ".join(dockerignore_rules())  # tylko reguły, bez komentarzy
        self.assertNotIn(cfg.DATA_PATH.name, rules)
        self.assertNotIn(".csv", rules)
        self.assertNotIn(cfg.DATA_PATH.name, DOCKERFILE)


class DockerfileTest(unittest.TestCase):
    def test_copies_only_allowed_paths(self):
        self.assertTrue(copy_sources())
        self.assertLessEqual(set(copy_sources()), ALLOWED)

    def test_uses_python_3_12_like_the_project(self):
        self.assertRegex(DOCKERFILE, r"(?m)^FROM python:3\.12-slim$")

    def test_runs_as_non_root_user(self):
        users = re.findall(r"(?m)^USER (\S+)$", DOCKERFILE)
        self.assertTrue(users)
        self.assertNotIn(users[-1], ("root", "0"))

    def test_has_healthcheck_on_streamlit_health_endpoint(self):
        self.assertIn("HEALTHCHECK", DOCKERFILE)
        self.assertIn("/_stcore/health", DOCKERFILE)

    def test_starts_the_streamlit_app_on_the_exposed_port(self):
        self.assertIn("EXPOSE 8501", DOCKERFILE)
        cmd = re.search(r"(?m)^CMD (.*)$", DOCKERFILE).group(1)
        self.assertIn("app/streamlit_app.py", cmd)
        self.assertIn("--server.port=8501", cmd)

    def test_installs_only_app_requirements_not_the_ml_stack(self):
        self.assertIn("pip install -r requirements-app.txt", DOCKERFILE)
        self.assertNotIn("pip install -r requirements.txt", DOCKERFILE)
        self.assertNotIn("libgomp1", DOCKERFILE)  # OpenMP był potrzebny tylko LightGBM

    def test_requirements_are_installed_before_the_code_is_copied(self):
        self.assertLess(DOCKERFILE.index("pip install -r requirements-app.txt"), DOCKERFILE.index("COPY --chown=appuser:appuser src"))


def pins(filename: str) -> dict:
    """Zależności z pliku requirements: {pakiet: wersja}; komentarze pomijamy."""
    lines = [l.strip() for l in (cfg.ROOT / filename).read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    return dict(line.split("==") for line in lines)


class RequirementsTest(unittest.TestCase):
    def test_every_dependency_is_pinned(self):
        for filename in ("requirements.txt", "requirements-app.txt"):
            lines = [l.strip() for l in (cfg.ROOT / filename).read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
            self.assertTrue(lines, filename)
            self.assertTrue(all("==" in line for line in lines), (filename, lines))

    def test_app_requirements_are_a_subset_with_the_same_versions(self):
        full, app = pins("requirements.txt"), pins("requirements-app.txt")
        for package, version in app.items():
            self.assertEqual(full.get(package), version, package)

    def test_app_requirements_do_not_contain_the_ml_stack(self):
        self.assertEqual(set(pins("requirements-app.txt")) & {"statsmodels", "scikit-learn", "lightgbm", "matplotlib"}, set())


if __name__ == "__main__":
    unittest.main()
