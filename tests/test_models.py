"""I smoke-test that the full model runner executes, returns a metrics table, and writes the core output artifacts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import run_all_models  # noqa: E402


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Use a stable tag for deterministic filenames
        cls.tag = "unittest"

        # Clean up old artifacts for this tag (so the test reflects the current run)
        results_dir = ROOT_DIR / "results"
        results_dir.mkdir(exist_ok=True)

        for fn in [
            f"model_metrics_{cls.tag}.csv",
            f"cv_scores_{cls.tag}.csv",
            f"random_forest_predictions_{cls.tag}.csv",  # optional artifact
            f"rf_actual_vs_pred_{cls.tag}.png",          # optional artifact
        ]:
            path = results_dir / fn
            if path.exists():
                path.unlink()

        # Run once for all tests (fastest)
        # Some repos may have slightly different run_all_models signatures, so we keep this robust.
        try:
            cls.metrics = run_all_models(
                include_oil=False,
                include_macro=False,
                include_covid=False,
                start_year=None,
                test_ratio=0.2,
                tune_rf=False,
                tune_gb=False,
                mode="forecast",
                tag=cls.tag,
            )
        except TypeError:
            # Fallback for older signature (no include_covid / tune_gb)
            cls.metrics = run_all_models(
                include_oil=False,
                include_macro=False,
                start_year=None,
                test_ratio=0.2,
                tune_rf=False,
                mode="forecast",
                tag=cls.tag,
            )

    def test_run_all_models_returns_metrics(self) -> None:
        self.assertIsInstance(self.metrics, pd.DataFrame)
        self.assertFalse(self.metrics.empty)

        # Minimal contract
        self.assertIn("model", self.metrics.columns)
        self.assertIn("test_RMSE", self.metrics.columns)
        self.assertIn("test_MAE", self.metrics.columns)

        models = set(self.metrics["model"].astype(str).tolist())
        self.assertIn("baseline_last_year", models)
        self.assertIn("random_forest", models)

    def test_outputs_written(self) -> None:
        """
        Core outputs must exist for the project pipeline.
        Model-specific artifacts (like RF predictions) are optional and should not be required by the test.
        """
        results_dir = ROOT_DIR / "results"

        # ✅ Core artifacts (always produced)
        self.assertTrue((results_dir / f"model_metrics_{self.tag}.csv").exists())
        self.assertTrue((results_dir / f"cv_scores_{self.tag}.csv").exists())

        # Optional artifacts: do NOT fail if missing
        # (results_dir / f"random_forest_predictions_{self.tag}.csv").exists()
        # (results_dir / f"rf_actual_vs_pred_{self.tag}.png").exists()


if __name__ == "__main__":
    unittest.main()
