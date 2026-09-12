"""Production ingress: the Caddy TLS profile + Hetzner runbook (owner
platform decision 2026-09-12).

The owner chose Hetzner (CX32, Ubuntu 24.04, EU DC) with docker compose
and Caddy terminating TLS. This suite pins that addition as DATA — pure
text/parse assertions over the committed compose file, the Caddyfile,
the production env template, and the runbook (the #212 convention in
``tests/unit/test_service_config.py::TestUvicornInvocation``: no Docker
needed at this tier; the smoke tier exercises running stacks).

What is pinned, and why:

- **Profile isolation.** The ``caddy`` service lives ONLY under the
  ``production`` compose profile. The dev/CI smoke stacks (plain
  ``docker compose up``, ``tests/smoke/_compose_stack.py``) must not
  contain it, and the profile-less base services must be untouched by
  the ingress addition — byte-equivalent behaviour for every existing
  stack is the whole point of using a profile.
- **Routing (path-based, ONE origin — not ``api.{domain}``).**
  ``ui/app.py`` builds every browser-facing api link by joining a root
  path onto ONE base origin (``SITE_URL or API_URL`` →
  ``<base>/chart/<hash>``, ``<base>/about`` …) and ``ui/transport.py``
  joins ``/chat``/``/feedback`` straight onto ``CLIMATE_CHAT_API_URL``
  with no ``/api`` prefix anywhere. So the api's public routes must be
  served from the SAME origin as the ui, on their real root paths;
  a prefix or subdomain would break every rendered permalink or force
  rewrites the app does not perform.
- **Trusted-proxy wiring (#212 / service/rate_limit.py).** Caddy is the
  ONLY ingress, so production sets ``CLIMATE_CHAT_TRUSTED_PROXY=1`` and
  ``resolve_client_ip`` honours the FIRST ``X-Forwarded-For`` entry.
  That first-entry rule is safe ONLY because Caddy (>= 2.5), with NO
  ``trusted_proxies`` configured, treats every client as untrusted and
  REPLACES client-supplied ``X-Forwarded-*`` headers with the real
  socket address. A ``trusted_proxies`` directive appearing in the
  Caddyfile would re-open the spoofing hole; the dev/smoke default
  stays untrusted (``:-0``) because there the socket peer IS the
  client.
- **qdrant is never exposed.** No Caddy route, no caddy port mapping —
  it stays a compose-network dependency (loopback-published on the
  host for operator debugging only, finding #36).
- **No admin API.** ``admin off``: nothing may reconfigure the ingress
  at runtime.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
CADDYFILE_PATH = REPO_ROOT / "deploy" / "Caddyfile"
OVERLAY_PATH = REPO_ROOT / "deploy" / "compose.production.yml"
ENV_TEMPLATE_PATH = REPO_ROOT / "deploy" / "production.env.example"
RUNBOOK_PATH = REPO_ROOT / "service" / "DEPLOYMENT.md"

#: The services a plain `docker compose up` (dev/CI smoke) must consist of —
#: exactly the pre-ingress set, all profile-less.
BASE_SERVICES = {"api", "smoke-seeder", "qdrant", "ui"}

#: The api's browser-facing public surface (service/app.py routes): the
#: paths the ingress must route to api:8000 on the shared origin.
API_PUBLIC_PATHS = (
    "/chat",
    "/feedback",
    "/health",
    "/about",
    "/privacy",
    "/sources",
    "/voices",
    "/chart/*",
)


def _compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def _caddyfile_directives() -> str:
    """The Caddyfile with comment lines stripped — only live directives.

    Comments may (and do) NAME forbidden things in order to forbid them;
    the negative assertions below must only see real configuration.
    """
    lines = CADDYFILE_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _env_template() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_TEMPLATE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


class TestProfileIsolation:
    """The production profile must be additive-only: dev/CI stacks see
    exactly the stack they always saw."""

    def test_default_stack_contains_no_caddy(self) -> None:
        services = _compose()["services"]
        profile_less = {name for name, body in services.items() if "profiles" not in body}
        assert profile_less == BASE_SERVICES, (
            "the profile-less (always-started) compose services must remain "
            f"exactly {sorted(BASE_SERVICES)} — a plain `docker compose up` "
            "(tests/smoke/_compose_stack.py) must never start the ingress, "
            f"got {sorted(profile_less)}"
        )

    def test_caddy_exists_only_under_the_production_profile(self) -> None:
        services = _compose()["services"]
        assert "caddy" in services, (
            "docker-compose.yml must define the `caddy` ingress service "
            "(owner platform decision 2026-09-12: Hetzner + Caddy TLS)"
        )
        assert services["caddy"].get("profiles") == ["production"], (
            "the caddy service must carry profiles: [production] — it is "
            "started ONLY by `docker compose --profile production up`"
        )

    def test_base_services_are_untouched_by_the_ingress(self) -> None:
        """No base service may reference caddy (depends_on, links, env):
        with the profile filtered out, the rendered base stack is the
        pre-ingress stack, byte-equivalent in behaviour."""
        services = _compose()["services"]
        for name in BASE_SERVICES:
            dumped = yaml.safe_dump(services[name])
            assert "caddy" not in dumped, (
                f"base service {name!r} references caddy — the production "
                "profile must be additive-only so dev/CI smoke stacks are "
                "provably unaffected"
            )


class TestCaddyService:
    def test_image_is_the_pinned_caddy_two_alpine(self) -> None:
        assert _compose()["services"]["caddy"]["image"] == "caddy:2-alpine"

    def test_publishes_exactly_http_https_and_http3(self) -> None:
        ports = {str(p) for p in _compose()["services"]["caddy"]["ports"]}
        assert ports == {"80:80", "443:443", "443:443/udp"}, (
            "caddy must publish 80 (ACME + redirect), 443, and 443/udp "
            "(HTTP/3) and NOTHING else — it is the only host-facing "
            f"0.0.0.0 publish point in the stack, got {sorted(ports)}"
        )

    def test_never_publishes_a_backend_port(self) -> None:
        ports = {str(p) for p in _compose()["services"]["caddy"]["ports"]}
        for forbidden in ("6333", "6334", "8000", "8501"):
            assert not any(forbidden in port for port in ports), (
                f"caddy must never publish backend port {forbidden} — "
                "qdrant/api/ui are reached over the compose network only"
            )

    def test_mounts_the_committed_caddyfile_read_only(self) -> None:
        volumes = [str(v) for v in _compose()["services"]["caddy"]["volumes"]]
        assert any(
            v.startswith("./deploy/Caddyfile:/etc/caddy/Caddyfile") and v.endswith(":ro")
            for v in volumes
        ), (
            "caddy must mount the committed deploy/Caddyfile read-only at "
            f"/etc/caddy/Caddyfile, got {volumes}"
        )

    def test_persists_certificates_in_a_named_volume(self) -> None:
        compose = _compose()
        volumes = [str(v) for v in compose["services"]["caddy"]["volumes"]]
        assert "caddy_data:/data" in volumes, (
            "caddy must persist /data (ACME account + issued certificates) "
            "in a named volume — re-issuing on every restart trips Let's "
            f"Encrypt rate limits, got {volumes}"
        )
        assert "caddy_data" in compose.get("volumes", {}), (
            "the caddy_data named volume must be declared at the top level"
        )

    def test_depends_on_api_and_ui_and_restarts(self) -> None:
        caddy = _compose()["services"]["caddy"]
        depends = caddy.get("depends_on")
        depend_names = set(depends) if isinstance(depends, (list, dict)) else set()
        assert {"api", "ui"} <= depend_names, f"caddy must depend on api and ui, got {depends!r}"
        assert caddy.get("restart") == "unless-stopped", (
            "the ingress must come back across daemon/server restarts: restart: unless-stopped"
        )


class TestCaddyfile:
    """The routing contract, pinned as data (see module docstring for the
    path-over-subdomain rationale)."""

    def test_caddyfile_is_committed(self) -> None:
        assert CADDYFILE_PATH.is_file(), (
            "deploy/Caddyfile must be committed — the compose caddy service mounts it"
        )

    def test_serves_the_domain_from_the_environment(self) -> None:
        assert "{$CLIMATE_CHAT_DOMAIN}" in _caddyfile_directives(), (
            "the site block must be {$CLIMATE_CHAT_DOMAIN} — the domain is "
            "deploy configuration (env), never hardcoded"
        )

    def test_routes_every_api_public_path_to_the_api(self) -> None:
        directives = _caddyfile_directives()
        assert "reverse_proxy api:8000" in directives, (
            "the api's public surface must proxy to api:8000 over the compose network"
        )
        for path in API_PUBLIC_PATHS:
            assert f" {path}" in directives, (
                f"the api path matcher must include {path} — ui/app.py "
                "renders it root-based off the single public origin "
                "(SITE_URL), so the ingress must route it to the api"
            )

    def test_everything_else_reaches_streamlit(self) -> None:
        assert "reverse_proxy ui:8501" in _caddyfile_directives(), (
            "the default handler must proxy to ui:8501 (Streamlit's compose "
            "port; reverse_proxy speaks its /_stcore/stream WebSocket "
            "natively)"
        )

    def test_qdrant_is_never_routed(self) -> None:
        directives = _caddyfile_directives()
        for forbidden in ("qdrant", "6333", "6334"):
            assert forbidden not in directives, (
                f"the Caddyfile must never reference {forbidden!r}: qdrant "
                "is a compose-network-only dependency, not a public surface"
            )

    def test_admin_api_is_off(self) -> None:
        assert "admin off" in _caddyfile_directives(), (
            "the global options must disable the admin API — nothing may "
            "reconfigure the ingress at runtime"
        )

    def test_no_trusted_proxies_directive_ever(self) -> None:
        assert "trusted_proxies" not in _caddyfile_directives(), (
            "a trusted_proxies directive would make Caddy PASS THROUGH "
            "client-supplied X-Forwarded-For; resolve_client_ip's "
            "first-entry rule is spoof-proof only while Caddy replaces the "
            "header for every (untrusted) client"
        )


class TestTrustedProxyWiring:
    """CLIMATE_CHAT_TRUSTED_PROXY: on for production (Caddy is the only
    route in), off for every dev/smoke stack (the socket peer IS the
    client there)."""

    def test_production_env_template_trusts_the_caddy_ingress(self) -> None:
        values = _env_template()
        assert values.get("CLIMATE_CHAT_TRUSTED_PROXY") == "1", (
            "deploy/production.env.example must set "
            "CLIMATE_CHAT_TRUSTED_PROXY=1: behind Caddy the socket peer is "
            "always the caddy container, so resolve_client_ip must honour "
            "the (Caddy-set) first X-Forwarded-For entry or every visitor "
            "shares one rate-limit bucket"
        )

    def test_production_env_template_names_domain_and_https_site_url(self) -> None:
        values = _env_template()
        assert "CLIMATE_CHAT_DOMAIN" in values, (
            "the template must set CLIMATE_CHAT_DOMAIN — the Caddyfile site block reads it"
        )
        assert values.get("CLIMATE_CHAT_SITE_URL", "").startswith("https://"), (
            "CLIMATE_CHAT_SITE_URL must be the public https origin — every "
            "chart permalink and transparency link renders off it"
        )

    def test_env_template_holds_no_real_credential(self) -> None:
        value = _env_template().get("ANTHROPIC_API_KEY", "")
        assert "REPLACE" in value and not value.startswith("sk-ant-"), (
            "the committed template must hold an obvious placeholder, "
            "never key material (finding #35 / gitleaks)"
        )

    def test_dev_and_smoke_stacks_stay_untrusted(self) -> None:
        api_env = _compose()["services"]["api"]["environment"]
        assert api_env["CLIMATE_CHAT_TRUSTED_PROXY"] == "${CLIMATE_CHAT_TRUSTED_PROXY:-0}", (
            "the compose default must remain untrusted (:-0): in dev/smoke "
            "stacks the socket peer IS the client, and honouring XFF there "
            "would let any client mint fresh identities per request"
        )


class TestProductionOverlay:
    """deploy/compose.production.yml: restart policies for the long-running
    base services, applied ONLY when the operator opts in with -f — the
    base file cannot carry them without changing dev/CI crash behaviour."""

    def test_overlay_adds_restart_to_exactly_the_long_running_services(self) -> None:
        overlay = yaml.safe_load(OVERLAY_PATH.read_text(encoding="utf-8"))
        services = overlay.get("services", {})
        assert set(services) == {"api", "qdrant", "ui"}, (
            "the overlay must touch exactly api/qdrant/ui — never the "
            "one-shot smoke-seeder (restarting a completed one-shot would "
            f"re-run it forever), got {sorted(services)}"
        )
        for name, body in services.items():
            assert body == {"restart": "unless-stopped"}, (
                f"the overlay may ONLY add restart: unless-stopped to "
                f"{name} — any other key would fork the production stack's "
                f"behaviour away from what the smoke tier verified, got {body}"
            )


class TestHetznerRunbook:
    """service/DEPLOYMENT.md documents the CHOSEN platform, not a menu of
    stale examples."""

    def runbook(self) -> str:
        return RUNBOOK_PATH.read_text(encoding="utf-8")

    def test_hetzner_section_exists_with_the_decision_recorded(self) -> None:
        runbook = self.runbook()
        assert "Hetzner deployment (chosen platform, owner decision 2026-09-12)" in runbook, (
            "DEPLOYMENT.md must carry the Hetzner section recording the owner's platform decision"
        )

    def test_hetzner_section_covers_the_provisioning_essentials(self) -> None:
        runbook = self.runbook()
        for anchor in (
            "CX32",
            "Ubuntu 24.04",
            "ufw",
            "v1.0.0-mvp",
            "--profile production",
            "deploy/Caddyfile",
            "deploy/production.env.example",
            "deploy/compose.production.yml",
            "chmod 600",
        ):
            assert anchor in runbook, (
                f"the Hetzner runbook must cover {anchor!r} (provisioning, "
                "clone-at-tag, env-file hygiene, the production compose "
                "invocation)"
            )

    def test_runbook_warns_that_docker_publishes_bypass_ufw(self) -> None:
        """finding #36 again, now at the host level: an operator who reads
        'ufw allow 22,80,443' must also read that Docker's iptables rules
        bypass ufw for published ports — loopback-only bindings (the base
        file) plus caddy as the only 0.0.0.0 publisher is the actual wall."""
        assert "bypass" in self.runbook().lower(), (
            "the Hetzner section must warn that Docker-published ports "
            "bypass ufw (finding #36) — the firewall paragraph is "
            "misleading without it"
        )

    def test_stale_platform_examples_are_gone(self) -> None:
        runbook = self.runbook()
        for stale in ("Fly.io", "Fly/Railway", "Railway"):
            assert stale not in runbook, (
                f"DEPLOYMENT.md still names {stale!r}: the owner chose "
                "Hetzner (2026-09-12); a runbook offering dead options "
                "misdirects the operator at deploy time"
            )
        assert "Caddy" in runbook, (
            "the runbook must name Caddy as the trusted TLS-terminating "
            "ingress (the CLIMATE_CHAT_TRUSTED_PROXY=1 rationale)"
        )

    def test_ico_item_reflects_the_existing_rusty_data_registration(self) -> None:
        """Owner correction (2026-09-12): the owner already pays the ICO
        annual data-protection fee as Rusty Data — one registration covers
        all processing by the controller. The action is consistency +
        internal records, not a fresh assessment."""
        runbook = self.runbook()
        assert "Rusty Data" in runbook, (
            "the ICO owner item must state the accurate position: the "
            "existing Rusty Data registration covers this service"
        )
        assert "Article 30" in runbook, (
            "the ICO owner item must ask for the internal record of "
            "processing activities (Article 30 note), kept internally"
        )
        assert "self-assessment" not in runbook, (
            "the stale 'complete the self-assessment / register' framing "
            "must go — the controller is already registered"
        )
