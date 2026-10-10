"""Tiny Phase 4 correctness/CLI/export check; never launches pilot or study."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from phase4_training_proxy import VARIANTS, parse_args, run


DIAGNOSTIC_FIELDS = {
    "max_abs_error_any_check", "max_relative_error_any_check",
    "max_abs_error_over_reference_max", "relative_l2_error_table", "failed_check_names",
}
CHECK_DIAGNOSTIC_FIELDS = {
    "reference_max_abs", "reference_l2_norm", "error_l2_norm", "relative_l2_error",
    "max_abs_error_over_reference_max",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    script = Path(__file__).with_name("phase4_training_proxy.py")
    help_result = subprocess.run([sys.executable, str(script), "--help"],
                                 capture_output=True, text=True, timeout=60)
    require(help_result.returncode == 0 and "--profile" in help_result.stdout,
            f"CLI help failed: {help_result.stderr}")
    available = torch.cuda.is_available()
    # Always exercise CPU fallback; if present, also check the real CUDA paths.
    devices = ["cpu", "cuda"] if available else ["auto"]
    with tempfile.TemporaryDirectory(prefix="phase4_smoke_") as temporary:
        for device in devices:
            args = parse_args(["--profile", "smoke", "--device", device,
                               "--out-dir", str(Path(temporary) / device)])
            out, passed = run(args)
            require(passed, f"Validation failed on {device}")
            validation = json.loads((out / "phase4_validation.json").read_text())
            metadata = json.loads((out / "phase4_metadata.json").read_text())
            with (out / "phase4_summary.csv").open(newline="") as f:
                summary = list(csv.DictReader(f))
            with (out / "phase4_raw.csv").open(newline="") as f:
                raw = list(csv.DictReader(f))
            require(len(validation) == 3 and len(summary) == 9 and len(raw) == 36,
                    "Unexpected artifact row counts")
            require({r["variant"] for r in summary} == set(VARIANTS), "Missing variants")
            expected_device = f"cuda:{torch.cuda.current_device()}" if device == "cuda" else "cpu"
            require(metadata["device"] == expected_device,
                    "Device selection/fallback failed")
            for config in validation:
                require(config["validation_passed"], "Invalid configuration")
                for variant in config["variants"]:
                    require(DIAGNOSTIC_FIELDS <= variant.keys(), "Missing JSON aggregate diagnostics")
                    require(variant["failed_check_names"] == "", "Unexpected failed checks")
                    require(all(CHECK_DIAGNOSTIC_FIELDS <= c.keys() for c in variant["checks"].values()),
                            "Missing per-check diagnostics")
                    require(all(c["passed"] and c["shape_ok"] and c["finite"]
                                for c in variant["checks"].values()), "Gradient/shape check failed")
            for row in summary:
                require(DIAGNOSTIC_FIELDS <= row.keys(), "Missing summary diagnostics")
                require(row["failed_check_names"] == "", "Unexpected summary failed checks")
                require(row["validation_passed"] == "True", "Failed row in timing summary")
                require(float(row["proxy_step_median_ms"]) > 0 and
                        float(row["accumulation_region_median_ms"]) > 0, "Invalid timing")
        if not available:
            try:
                run(parse_args(["--profile", "smoke", "--device", "cuda"]))
            except RuntimeError as exc:
                require("CUDA requested but unavailable" in str(exc), "Unexpected CUDA error")
            else:
                raise AssertionError("Explicit CUDA request must fail when unavailable")
    print("PASS: CLI help, device handling, all variants, validation, and CSV/JSON exports.")
    print("Smoke artifacts used temporary directories and were removed; no study was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
