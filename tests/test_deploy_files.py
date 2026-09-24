"""Testy plików wdrożeniowych: konfiguracja compose, Caddyfile oraz brak sekretów i adresów IP w repozytorium."""
import json
import os
import re
import shutil
import subprocess
import unittest

import config as cfg

COMPOSE = cfg.ROOT / "docker-compose.yml"
CADDYFILE = (cfg.ROOT / "deploy" / "Caddyfile").read_text(encoding="utf-8")
HAVE_COMPOSE = shutil.which("docker") is not None and subprocess.run(
    ["docker", "compose", "version"], capture_output=True
).returncode == 0

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
ALLOWED_IPS = {"0.0.0.0", "127.0.0.1"}  # adres nasłuchu w kontenerze i pętla zwrotna, nie adresy serwera
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"ssh-(?:rsa|ed25519) AAAA"),
    re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{8,}"),
]
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".css", ".example", ""}


def compose_config() -> dict:
    env = {**os.environ, "DOMAIN": "forecast.example.com"}
    result = subprocess.run(["docker", "compose", "-f", str(COMPOSE), "config", "--format", "json"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@unittest.skipUnless(HAVE_COMPOSE, "brak Docker Compose")
class ComposeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = compose_config()
        cls.app, cls.caddy = cls.config["services"]["app"], cls.config["services"]["caddy"]

    def test_app_does_not_publish_any_port(self):
        self.assertNotIn("ports", self.app)
        self.assertIn("8501", [str(p) for p in self.app.get("expose", [])])

    def test_only_proxy_publishes_http_and_https(self):
        self.assertEqual(sorted({int(p["target"]) for p in self.caddy["ports"]}), [80, 443])

    def test_app_container_is_hardened(self):
        self.assertTrue(self.app["read_only"])
        self.assertEqual(self.app["cap_drop"], ["ALL"])
        self.assertIn("no-new-privileges:true", self.app["security_opt"])

    def test_proxy_waits_for_healthy_app(self):
        self.assertEqual(self.caddy["depends_on"]["app"]["condition"], "service_healthy")

    def test_certificates_live_in_a_named_volume(self):
        self.assertIn("caddy_data", self.config["volumes"])
        self.assertIn("/data", [v["target"] for v in self.caddy["volumes"]])

    def test_domain_is_required(self):
        env = {k: v for k, v in os.environ.items() if k != "DOMAIN"}
        result = subprocess.run(["docker", "compose", "-f", str(COMPOSE), "config", "-q"], capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DOMAIN", result.stderr)


class CaddyfileTest(unittest.TestCase):
    def test_domain_comes_from_environment_and_app_is_proxied(self):
        self.assertIn("{$DOMAIN}", CADDYFILE)
        self.assertIn("reverse_proxy app:8501", CADDYFILE)

    def test_security_headers_are_set(self):
        for header in ("Strict-Transport-Security", "X-Content-Type-Options", "Referrer-Policy", "-Server"):
            self.assertIn(header, CADDYFILE)

    def test_basic_auth_is_only_a_commented_example_without_credentials(self):
        active = [l for l in CADDYFILE.splitlines() if l.strip() and not l.strip().startswith("#")]
        self.assertFalse(any("basic_auth" in l for l in active))
        self.assertNotIn("$2a$", CADDYFILE)  # brak wpisanego hasza hasła


class EnvTest(unittest.TestCase):
    def test_env_file_is_ignored_and_example_has_placeholder_domain(self):
        self.assertIn(".env", (cfg.ROOT / ".gitignore").read_text().splitlines())
        example = (cfg.ROOT / ".env.example").read_text()
        self.assertRegex(example, r"(?m)^DOMAIN=\S+\.example\.com$")


def tracked_text_files() -> list:
    if shutil.which("git") is None or not (cfg.ROOT / ".git").exists():
        return []
    out = subprocess.run(["git", "ls-files"], cwd=cfg.ROOT, capture_output=True, text=True).stdout.split("\n")
    files = [cfg.ROOT / f for f in out if f and not f.startswith("outputs/") and not f.startswith("docs/")]
    return [f for f in files if f.suffix in TEXT_SUFFIXES and f.is_file() and f.name != "test_deploy_files.py"]


class NoSecretsInRepositoryTest(unittest.TestCase):
    """Zakaz z CLAUDE.md: żadnych haseł, kluczy ani adresów IP serwera w repozytorium."""

    @classmethod
    def setUpClass(cls):
        cls.files = tracked_text_files()

    def test_there_are_files_to_scan(self):
        if not (cfg.ROOT / ".git").exists():
            self.skipTest("brak repozytorium git")
        self.assertGreater(len(self.files), 20)

    def test_no_ip_addresses_other_than_loopback_and_listen_all(self):
        for path in self.files:
            found = set(IPV4.findall(path.read_text(encoding="utf-8", errors="ignore"))) - ALLOWED_IPS
            self.assertEqual(found, set(), f"{path.relative_to(cfg.ROOT)} zawiera adres IP")

    def test_no_secrets_or_private_keys(self):
        for path in self.files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                self.assertIsNone(pattern.search(text), f"{path.relative_to(cfg.ROOT)}: {pattern.pattern}")


if __name__ == "__main__":
    unittest.main()
