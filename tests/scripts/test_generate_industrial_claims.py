"""Tests for scripts/generate_industrial_claims.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_industrial_claims.py"

spec = importlib.util.spec_from_file_location("generate_industrial_claims", SCRIPT)
assert spec is not None and spec.loader is not None
gen = importlib.util.module_from_spec(spec)
sys.modules.setdefault("generate_industrial_claims", gen)
spec.loader.exec_module(gen)


def _artifact(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    base = {
        "schema": "OpenLithoHub.industrial-benchmark.v1",
        "kind": kind,
        "status": "SUCCESS",
        "git_commit": "abcdef123456",
        "timestamp_utc": "2026-09-20T00:00:00Z",
        "hardware": {"cpu_model": "Test CPU", "physical_ram_bytes": 48 * (1 << 30), "gpu": None},
        "software": {"python": "3.12"},
        "claim_scope": {"physics_claim": "NOT_FOUNDRY_CALIBRATED"},
        "reproducibility": {"command": "run"},
    }
    base.update(payload)
    return base


def _stat(median: float, n: int = 5) -> dict[str, float | int]:
    return {"n": n, "median": median, "p10": median, "p90": median, "min": median, "max": median}


@pytest.fixture
def runtime_artifact() -> dict[str, Any]:
    return _artifact(
        "runtime",
        {
            "memory": {
                "per_size": {
                    "4096": {
                        "dense_full": {"peak_rss_bytes_median": 1.0 * (1 << 30)},
                        "b04_selective": {"peak_rss_bytes_median": 0.4 * (1 << 30)},
                        "streaming_memory_reduction_pct": 60.0,
                    },
                    "8192": {
                        "dense_full": {"peak_rss_bytes_median": 2.0 * (1 << 30)},
                        "b04_selective": {"peak_rss_bytes_median": 1.9 * (1 << 30)},
                        "streaming_memory_reduction_pct": 5.0,
                    },
                },
                "max_streamed_size_px": 65536,
                "dense_structural_infeasible_sizes_px": [65536],
            },
            "comparisons": {
                "4096": {
                    "dense_vs_selective": {
                        "baseline_median_s": 0.1,
                        "candidate_median_s": 0.2,
                        "speedup": 0.5,
                        "metric": "execution_wall_s",
                    }
                }
            },
            "rows": [
                {
                    "mode": "b04_selective",
                    "size_px": [65536, 65536],
                    "peak_rss_bytes": _stat(0.5 * (1 << 30)),
                }
            ],
            "policy": {"repeats": 5},
        },
    )


@pytest.fixture
def quality_artifact() -> dict[str, Any]:
    agg = {
        "dummy-identity": {
            "pvband_mean_nm": _stat(10.0),
            "mrc_violation_rate": _stat(0.10),
            "wafer_epe_mean_nm": _stat(8.0),
        },
        "levelset-ilt": {
            "pvband_mean_nm": _stat(7.0),
            "mrc_violation_rate": _stat(0.01),
            "wafer_epe_mean_nm": _stat(6.0),
            "optimization_wall_s": _stat(70.0),
        },
        "surrogate-ilt": {
            "pvband_mean_nm": _stat(9.0),
            "mrc_violation_rate": _stat(0.02),
            "wafer_epe_mean_nm": _stat(9.9),
            "optimization_wall_s": _stat(10.0),
        },
    }
    return _artifact(
        "quality",
        {
            "datasets": {
                "sky130hd-ibex-tiles": {
                    "description": "real routed sky130hd ibex tiles",
                    "policy": {"max_tiles": 6, "tile_size_px": 1024, "ilt_iterations": 50},
                    "aggregate": agg,
                    "comparisons": {
                        "identity_vs_levelset": {
                            "pvband_mean_nm": {
                                "metric": "pvband_mean_nm",
                                "baseline": "dummy-identity",
                                "candidate": "levelset-ilt",
                                "baseline_median": 10.0,
                                "candidate_median": 7.0,
                                "reduction_pct": 30.0,
                            },
                            "mrc_violation_rate": {
                                "metric": "mrc_violation_rate",
                                "baseline": "dummy-identity",
                                "candidate": "levelset-ilt",
                                "baseline_median": 0.10,
                                "candidate_median": 0.01,
                                "reduction_pct": 90.0,
                            },
                            "wafer_epe_mean_nm": {
                                "metric": "wafer_epe_mean_nm",
                                "baseline": "dummy-identity",
                                "candidate": "levelset-ilt",
                                "baseline_median": 8.0,
                                "candidate_median": 6.0,
                                "reduction_pct": 25.0,
                            },
                        },
                        "levelset_vs_surrogate_runtime": {
                            "baseline": "levelset-ilt",
                            "candidate": "surrogate-ilt",
                            "baseline_median_s": 70.0,
                            "candidate_median_s": 10.0,
                            "matched_iterations": 50,
                        },
                    },
                }
            }
        },
    )


@pytest.fixture
def fulldie_artifact() -> dict[str, Any]:
    return _artifact(
        "full_die",
        {
            "die_size_px": [555355, 555355],
            "tile_px": 32768,
            "grid_tiles_per_side": 17,
            "n_grid_tiles": 289,
            "n_sample_tiles": 5,
            "sample_rows": [{"screen_query_wall_s": 1.0, "screened_fraction": 0.8}] * 5,
            "sample_screen_query_wall_s_mean": 1.0,
            "sampled_screened_fraction_range": [0.07, 0.92],
            "full_die_survey_wall_s_estimate": {
                "value": 289.0,
                "method": "mean",
                "status": "ESTIMATE_NOT_MEASUREMENT",
            },
            "dense_die_raster_bytes_structural": 1233676704100,
            "dense_die_status": "INFEASIBLE_ON_REFERENCE_MACHINE",
            "physical_ram_bytes": 48 * (1 << 30),
            "known_scaling_limit": "row index cost",
        },
    )


class TestDeriveClaims:
    def test_memory_claims_threshold(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        by_id = {c["claim_id"]: c for c in claims}
        assert by_id["IB-MEM-4096"]["headline"] is True
        assert by_id["IB-MEM-4096"]["value"] == "60.0%"
        # 5% reduction is below the 20% headline threshold
        assert by_id["IB-MEM-8192"]["headline"] is False

    def test_runtime_claim_scoped_when_dense_wins(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        rt = next(c for c in claims if c["claim_id"] == "IB-RT-4096")
        assert rt["value"] == "0.50x"
        assert rt["headline"] is False
        assert "dense is faster" in rt["scope"]

    def test_scale_claim_always_headline(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        scale = next(c for c in claims if c["claim_id"] == "IB-SCALE-65536")
        assert scale["headline"] is True
        assert "65536x65536" in scale["value"]

    def test_quality_claims_threshold(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        by_id = {c["claim_id"]: c for c in claims}
        # P0.8: quality comparisons are facts only — never headline in v1
        assert by_id["IB-Q-ILT-PVB"]["headline"] is False
        assert by_id["IB-Q-ILT-MRC"]["headline"] is False
        assert by_id["IB-Q-ILT-WEPE"]["headline"] is False
        # values carry absolute + relative change
        assert "10.00000 -> 7.00000" in by_id["IB-Q-ILT-PVB"]["value"]
        assert "30.0%" in by_id["IB-Q-ILT-PVB"]["value"]
        # rule-based-opc is absent from this fixture's aggregate: no RB claims
        assert not any(c["claim_id"].startswith("IB-Q-RB") for c in claims)

    def test_surrogate_claim_scoped_when_quality_degrades(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        surr = next(c for c in claims if c["claim_id"] == "IB-Q-SURR")
        # 7x faster but wafer EPE degrades 8.0 -> 9.9 (>10%) -> not headline
        assert surr["value"] == "7.00x"
        assert surr["headline"] is False
        assert "NOT quality-normalized" in surr["scope"]

    def test_surrogate_claim_headline_when_quality_holds(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        quality_artifact["datasets"]["sky130hd-ibex-tiles"]["aggregate"]["surrogate-ilt"][
            "wafer_epe_mean_nm"
        ] = _stat(6.5)
        quality_artifact["datasets"]["sky130hd-ibex-tiles"]["aggregate"]["surrogate-ilt"][
            "pvband_mean_nm"
        ] = _stat(7.2)
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        surr = next(c for c in claims if c["claim_id"] == "IB-Q-SURR")
        # quality holds at this budget, but the v1 quality-headline flag is
        # off: surrogate runtime stays a scoped fact, never a headline.
        assert surr["headline"] is False
        assert "quality within tolerance" in surr["scope"]

    def test_die_structural_claim(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        die = next(c for c in claims if c["claim_id"] == "IB-DIE-1")
        assert "TB" in die["value"]
        assert die["headline"] is True

    def test_empty_artifacts_safe(self) -> None:
        assert gen.derive_claims({}, {}) == []


class TestCheckReadme:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "README.md"
        path.write_text(text, encoding="utf-8")
        return path

    def _claim(self, claim_id: str, value: str, headline: bool) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "value": value,
            "headline": headline,
            "metric": "m",
            "scope": "s",
        }

    def test_ok(self, tmp_path) -> None:
        readme = self._write(tmp_path, "# x\n`IB-MEM-4096` shows 60.0%\n")
        claims = [self._claim("IB-MEM-4096", "60.0%", True)]
        assert gen.check_readme(claims, readme) == []

    def test_unknown_id(self, tmp_path) -> None:
        readme = self._write(tmp_path, "`IB-NOPE` says 1.0%\n")
        claims = [self._claim("IB-MEM-4096", "60.0%", True)]
        problems = gen.check_readme(claims, readme)
        assert any("unknown claim id" in p for p in problems)

    def test_non_headline_mention_allowed(self, tmp_path) -> None:
        """Facts may be referenced in prose without quoting the value;
        only headline claims carry the exact-value requirement."""
        readme = self._write(tmp_path, "`IB-MEM-8192` is a measured fact\n")
        claims = [self._claim("IB-MEM-8192", "5.0%", False)]
        assert gen.check_readme(claims, readme) == []

    def test_missing_exact_value(self, tmp_path) -> None:
        readme = self._write(tmp_path, "`IB-MEM-4096` reduces memory a lot\n")
        claims = [self._claim("IB-MEM-4096", "60.0%", True)]
        problems = gen.check_readme(claims, readme)
        assert any("exact artifact value" in p for p in problems)


class TestRenderMarkdown:
    def test_contains_headline_and_firewall(
        self, runtime_artifact, quality_artifact, fulldie_artifact
    ) -> None:
        claims = gen.derive_claims(
            {"runtime": runtime_artifact, "quality": quality_artifact, "fulldie": fulldie_artifact},
            {},
        )
        text = gen.render_markdown(claims, {}, "Test CPU (CPU, 48 GB RAM)")
        assert "Headline-eligible claims" in text
        assert "Claim firewall" in text
        assert "No commercial-tool" in text
        assert "IB-MEM-4096" in text
