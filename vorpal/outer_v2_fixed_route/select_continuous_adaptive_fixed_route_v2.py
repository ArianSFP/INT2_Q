#!/usr/bin/env python3
"""Force the adaptive selector through isolated fixed-route codec v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


PINNED_SELECTOR_CORE_SHA256 = (
    "6d44da48e5b842c542e29fb36a4dea243eacbf917d0028b999e97131043fbbe0"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selector(path: Path | None = None):
    path = path or (
        Path(__file__).resolve().parent.parent / "select_continuous_adaptive.py"
    )
    path = path.resolve()
    actual_hash = sha256_path(path)
    if actual_hash != PINNED_SELECTOR_CORE_SHA256:
        raise ValueError(
            f"unaudited selector core {path}: {actual_hash} != "
            f"{PINNED_SELECTOR_CORE_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("_fixed_route_selector_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selector_dependency_bindings(selector) -> dict[str, str]:
    core_hash = sha256_path(Path(selector.__file__).resolve())
    if core_hash != PINNED_SELECTOR_CORE_SHA256:
        raise ValueError("selector core changed after audited import")
    return {
        "fixed_route_selector_wrapper_sha256": sha256_path(Path(__file__).resolve()),
        "delegated_selector_core_sha256": core_hash,
        "pinned_selector_core_sha256": PINNED_SELECTOR_CORE_SHA256,
    }


def validate_selector_dependency_bindings(
    bindings: object, expected: dict[str, str]
) -> None:
    if not isinstance(bindings, dict) or bindings != expected:
        raise ValueError("fixed-route selector dependency bindings mismatch")


def atomic_text(path: Path, payload: str) -> None:
    temporary = Path(str(path) + ".fixed-route.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def bind_selection_receipt(
    receipt_path: Path, bindings: dict[str, str]
) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format")
        != "continuous PLTE exact adaptive selection receipt fixed-route v2"
        or receipt.get("status") != "passed"
        or receipt.get("physical_selection_mode") != "fixed-route"
    ):
        raise ValueError("selector core did not emit a passed fixed-route v2 receipt")
    receipt["selector_dependency_bindings"] = bindings
    atomic_text(receipt_path, json.dumps(receipt, indent=2) + "\n")
    rebound = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_selector_dependency_bindings(
        rebound.get("selector_dependency_bindings"), bindings
    )
    checksum_path = receipt_path.with_name("selection.receipt.sha256")
    atomic_text(
        checksum_path,
        f"{sha256_path(receipt_path)}  {receipt_path.name}\n",
    )


def main() -> None:
    if "--physical-selection" in sys.argv:
        index = sys.argv.index("--physical-selection")
        if index + 1 >= len(sys.argv) or sys.argv[index + 1] != "fixed-route":
            raise ValueError("v2 selector only permits --physical-selection fixed-route")
    else:
        sys.argv.extend(["--physical-selection", "fixed-route"])
    selector = load_selector()
    bindings = selector_dependency_bindings(selector)
    core_stage_selection = selector.stage_selection

    def bound_stage_selection(args):
        receipt_path = core_stage_selection(args)
        bind_selection_receipt(receipt_path, bindings)
        return receipt_path

    selector.stage_selection = bound_stage_selection
    selector.main()


if __name__ == "__main__":
    main()
