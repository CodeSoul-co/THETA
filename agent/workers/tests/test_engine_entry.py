"""Regression: the agent worker directory must not shadow standard library modules."""
from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

WORKERS = Path(__file__).resolve().parents[1]
AGENT = WORKERS.parent


class EngineEntryImportPathTest(unittest.TestCase):
    def test_local_statistics_package_is_removed_from_the_engine_import_path(self) -> None:
        # The regression only appears when this directory is a top-level sys.path entry,
        # which is exactly how Python starts a script placed here.
        self.assertTrue((WORKERS / "statistics" / "__init__.py").is_file())
        program = textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from workers.engine_entry import harden_import_path
            sys.path.insert(0, str(Path("workers").resolve()))
            harden_import_path()
            assert str(Path("workers").resolve()) not in sys.path
            import statistics
            assert Path(statistics.__file__).resolve().parent != Path("workers").resolve(), statistics.__file__
            print(statistics.NormalDist(0.0, 1.0).pdf(0.0))
            """
        )
        result = subprocess.run([sys.executable, "-c", program], cwd=str(AGENT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(float(result.stdout.strip()), 0.3989422804014327, places=12)

    def test_engine_entry_keeps_the_agent_root_importable(self) -> None:
        from workers.engine_entry import harden_import_path  # noqa: F401  (import must not run the entry point)
        self.assertTrue((AGENT / "workers" / "api_overrides.py").is_file())


if __name__ == "__main__":
    unittest.main()
