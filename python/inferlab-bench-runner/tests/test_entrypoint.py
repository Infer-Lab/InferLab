import os
import subprocess
import sys
from pathlib import Path


def test_direct_file_entrypoint_resolves_the_staged_runner_package() -> None:
    package_root = Path(__file__).resolve().parents[1]
    runner_source = package_root / "src"
    measurement_source = package_root.parent / "inferlab-measurement-sdk" / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(runner_source), str(measurement_source)))
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            str(runner_source / "inferlab_bench_runner" / "bench_client.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
