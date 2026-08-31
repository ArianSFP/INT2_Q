#!/usr/bin/env python3
"""Run the exact source-free role-joint verifier on published repository files."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT = ROOT / "evaluation" / "qwen3_vorpal_role_joint_v1"
BASE = ROOT / "evaluation" / "qwen3_vorpal_v1"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required publication file is missing: {path}")
    return path


def main() -> None:
    core = require_file(HERE / "verify_joint_sparc4_source_free.py")
    with tempfile.TemporaryDirectory(prefix="vorpal-role-joint-") as directory:
        temporary = Path(directory)
        command = [
            sys.executable,
            str(core),
            "--wrapper",
            str(require_file(ARTIFACT / "vorpal_joint_sparc4.vjwrap")),
            "--base-bundle",
            str(require_file(BASE / "selected.wfouter")),
            "--manifest",
            str(require_file(BASE / "selected.manifest.json")),
            "--evaluation",
            str(require_file(BASE / "evaluation.json")),
            "--experiment-receipt",
            str(require_file(ARTIFACT / "joint_sparc4.receipt.json")),
            "--independent-verification",
            str(
                require_file(
                    ARTIFACT / "joint_sparc4.independent-verification.json"
                )
            ),
            "--encoder",
            str(require_file(HERE / "build_joint_sparc4.py")),
            "--full-verifier",
            str(require_file(HERE / "verify_joint_sparc4.py")),
            "--output",
            str(temporary / "source-free-verification.json"),
            "--tamper-output",
            str(temporary / "tamper-tests.json"),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
