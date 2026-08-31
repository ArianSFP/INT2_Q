#!/usr/bin/env python3
"""Build the deterministic Qwen3 PLTE broad-coverage evaluation manifest.

The selection reads safetensors headers and the identifiers in the published
evidence manifest, but never reads a weight payload.  Existing evidence blocks
are skipped so that the extension adds 400 new PLTE blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLTE = ROOT / "plte"
CHECKPOINT = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
BASE_PUBLICATION_COMMIT = "5d2a29cb60f2e068ac0a49cc33e04f51f515720e"
POLAR_UPSTREAM_COMMIT = "458187b9b03db1768a4b72d617e591f7862f6fca"
BLOCK_VALUES = 1 << 18
PROTOCOL_TAG = "PLTE-QWEN3-30B-A3B-COVERAGE-V1"
SELECTION_MATERIAL = (
    PROTOCOL_TAG.encode("utf-8")
    + b"\0"
    + REVISION.encode("ascii")
    + b"\0"
    + BASE_PUBLICATION_COMMIT.encode("ascii")
)
SELECTION_SEED = hashlib.sha256(SELECTION_MATERIAL).digest()
GLOBAL_STRATA = 32

ATTENTION_ROLES = (
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
EXPERT_ROLES = (
    "mlp.experts.{expert}.down_proj.weight",
    "mlp.experts.{expert}.gate_proj.weight",
    "mlp.experts.{expert}.up_proj.weight",
)
EXPECTED_NORMALIZED_ROLES = {
    "input_layernorm.weight",
    "lm_head.weight",
    "mlp.experts.{expert}.down_proj.weight",
    "mlp.experts.{expert}.gate_proj.weight",
    "mlp.experts.{expert}.up_proj.weight",
    "mlp.gate.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
}


@dataclass(frozen=True)
class TensorHeader:
    tensor: str
    shape: tuple[int, ...]
    dtype: str
    data_offsets: tuple[int, int]
    shard: str
    header_file: str
    header_length: int

    @property
    def values(self) -> int:
        return math.prod(self.shape)

    @property
    def full_blocks(self) -> int:
        return self.values // BLOCK_VALUES


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_role(tensor: str) -> tuple[int | None, str]:
    parts = tensor.split(".")
    if len(parts) >= 4 and parts[:2] == ["model", "layers"]:
        layer = int(parts[2])
        role = ".".join(parts[3:])
        if role.startswith("mlp.experts."):
            role_parts = role.split(".")
            role_parts[2] = "{expert}"
            role = ".".join(role_parts)
        return layer, role
    return None, tensor


def load_headers(directory: Path) -> tuple[dict[str, TensorHeader], str]:
    catalog: dict[str, TensorHeader] = {}
    digest = hashlib.sha256()
    files = sorted(directory.glob("model-*-of-*.safetensors.header.json"))
    if len(files) != 16:
        raise AssertionError(f"expected 16 header files, found {len(files)}")
    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_bytes(raw)))
        wrapper = json.loads(raw)
        header_length = int(wrapper["header_length"])
        shard = path.name.removesuffix(".header.json")
        for tensor, metadata in wrapper["header"].items():
            if tensor == "__metadata__":
                continue
            if tensor in catalog:
                raise AssertionError(f"duplicate tensor in headers: {tensor}")
            catalog[tensor] = TensorHeader(
                tensor=tensor,
                shape=tuple(int(x) for x in metadata["shape"]),
                dtype=str(metadata["dtype"]),
                data_offsets=tuple(int(x) for x in metadata["data_offsets"]),
                shard=shard,
                header_file=path.name,
                header_length=header_length,
            )
    return catalog, digest.hexdigest()


def selection_offset(label: str, population: int) -> tuple[int, str]:
    digest = hashlib.sha256(SELECTION_SEED + b"\0" + label.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % population, digest


def cyclic_pick(
    candidates: list[tuple[str, int]], excluded: set[tuple[str, int]], label: str
) -> tuple[tuple[str, int], str, int]:
    start, digest = selection_offset(label, len(candidates))
    for step in range(len(candidates)):
        candidate = candidates[(start + step) % len(candidates)]
        if candidate not in excluded:
            return candidate, digest, step
    raise AssertionError(f"all candidates excluded for {label}")


def source_range(header: TensorHeader, begin_value: int, values: int) -> tuple[int, int]:
    if header.dtype != "BF16":
        raise AssertionError(f"unexpected dtype for {header.tensor}: {header.dtype}")
    if begin_value < 0 or begin_value + values > header.values:
        raise AssertionError(f"source range outside tensor {header.tensor}")
    start = 8 + header.header_length + header.data_offsets[0] + 2 * begin_value
    return start, start + 2 * values - 1


def entry_for_block(
    header: TensorHeader,
    *,
    entry_id: str,
    role: str,
    layer: int | None,
    block_index: int,
    selection_digest: str,
    collision_steps: int,
    stratum_inventory_blocks: int,
    stratum: str,
) -> dict[str, object]:
    byte_range = source_range(header, block_index * BLOCK_VALUES, BLOCK_VALUES)
    return {
        "id": entry_id,
        "stratum": stratum,
        "layer": layer,
        "role": role,
        "tensor": header.tensor,
        "shape": list(header.shape),
        "dtype": header.dtype,
        "block_index": block_index,
        "tensor_full_blocks": header.full_blocks,
        "stratum_inventory_blocks": stratum_inventory_blocks,
        "selection_digest": selection_digest,
        "prior_evidence_collision_steps": collision_steps,
        "shard": header.shard,
        "header_file": header.header_file,
        "absolute_byte_range_in_shard": list(byte_range),
        "source_values": BLOCK_VALUES,
        "source_bytes": 2 * BLOCK_VALUES,
    }


def role_inventory(catalog: dict[str, TensorHeader]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for tensor, header in catalog.items():
        layer, role = normalized_role(tensor)
        row = grouped.setdefault(
            role,
            {"role": role, "tensors": 0, "layers": set(), "shapes": set()},
        )
        row["tensors"] = int(row["tensors"]) + 1
        if layer is not None:
            row["layers"].add(layer)  # type: ignore[union-attr]
        row["shapes"].add(header.shape)  # type: ignore[union-attr]
    if set(grouped) != EXPECTED_NORMALIZED_ROLES:
        raise AssertionError(
            f"unexpected role inventory: {sorted(set(grouped) ^ EXPECTED_NORMALIZED_ROLES)}"
        )
    output = []
    for role in sorted(grouped):
        row = grouped[role]
        output.append(
            {
                "role": role,
                "tensors": row["tensors"],
                "layers": sorted(row["layers"]),
                "shapes": [list(shape) for shape in sorted(row["shapes"])],
            }
        )
    return output


def build_manifest(
    catalog: dict[str, TensorHeader],
    header_bundle_sha256: str,
    evidence_path: Path,
) -> dict[str, object]:
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    expected_checkpoint = {"repo": CHECKPOINT, "revision": REVISION}
    if evidence["checkpoint"] != expected_checkpoint:
        raise AssertionError("published evidence uses an unexpected checkpoint")
    encoder_path = PLTE / "agent_root_polar_lattice_gate.py"
    decoder_path = PLTE / "agent_polar_codec_audit_independent_decoder.py"
    fetcher_path = PLTE / "agent_root_fetch_qwen_block.py"
    profile_path = PLTE / "agent_root_polar_escape_frozen_profiles.bin"
    encoder_sha256 = sha256_bytes(encoder_path.read_bytes())
    decoder_sha256 = sha256_bytes(decoder_path.read_bytes())
    fetcher_sha256 = sha256_bytes(fetcher_path.read_bytes())
    profile_sha256 = sha256_bytes(profile_path.read_bytes())
    if encoder_sha256 != evidence["encoder_sha256"]:
        raise AssertionError("encoder bytes differ from published evidence")
    if profile_sha256 != evidence["frozen_profile_sha256"]:
        raise AssertionError("frozen profile differs from published evidence")
    excluded = {
        (str(row["canonical_tensor"]), int(row["canonical_block_index"]))
        for row in evidence["reports"]
    }

    plte_blocks: list[dict[str, object]] = []
    for layer in range(48):
        for role in ATTENTION_ROLES:
            tensor = f"model.layers.{layer}.{role}"
            header = catalog[tensor]
            candidates = [(tensor, block) for block in range(header.full_blocks)]
            (selected_tensor, block), digest, steps = cyclic_pick(
                candidates, excluded, f"layer={layer}|role={role}"
            )
            assert selected_tensor == tensor
            short = role.removeprefix("self_attn.").removesuffix(".weight")
            plte_blocks.append(
                entry_for_block(
                    header,
                    entry_id=f"l{layer:02d}-attn-{short}-b{block}",
                    role=role,
                    layer=layer,
                    block_index=block,
                    selection_digest=digest,
                    collision_steps=steps,
                    stratum_inventory_blocks=header.full_blocks,
                    stratum=f"layer-{layer:02d}/{role}",
                )
            )

        for role_template in EXPERT_ROLES:
            role_label = role_template.replace("{expert}.", "")
            candidates: list[tuple[str, int]] = []
            for expert in range(128):
                tensor = f"model.layers.{layer}.{role_template.format(expert=expert)}"
                header = catalog[tensor]
                candidates.extend((tensor, block) for block in range(header.full_blocks))
            (tensor, block), digest, steps = cyclic_pick(
                candidates, excluded, f"layer={layer}|role={role_template}"
            )
            header = catalog[tensor]
            expert = int(tensor.split(".experts.", 1)[1].split(".", 1)[0])
            short = role_label.removeprefix("mlp.").removesuffix(".weight")
            plte_blocks.append(
                entry_for_block(
                    header,
                    entry_id=f"l{layer:02d}-expert{expert:03d}-{short}-b{block}",
                    role=role_template,
                    layer=layer,
                    block_index=block,
                    selection_digest=digest,
                    collision_steps=steps,
                    stratum_inventory_blocks=len(candidates),
                    stratum=f"layer-{layer:02d}/{role_template}",
                )
            )

    for tensor, short in (
        ("model.embed_tokens.weight", "embed"),
        ("lm_head.weight", "lmhead"),
    ):
        header = catalog[tensor]
        for stratum_index in range(GLOBAL_STRATA):
            lo = stratum_index * header.full_blocks // GLOBAL_STRATA
            hi = (stratum_index + 1) * header.full_blocks // GLOBAL_STRATA
            candidates = [(tensor, block) for block in range(lo, hi)]
            (_, block), digest, steps = cyclic_pick(
                candidates, excluded, f"tensor={tensor}|stratum={stratum_index}"
            )
            plte_blocks.append(
                entry_for_block(
                    header,
                    entry_id=f"{short}-s{stratum_index:02d}-b{block}",
                    role=tensor,
                    layer=None,
                    block_index=block,
                    selection_digest=digest,
                    collision_steps=steps,
                    stratum_inventory_blocks=hi - lo,
                    stratum=f"{tensor}/quantile-{stratum_index:02d}",
                )
            )

    router_blocks = []
    for layer in range(48):
        tensor = f"model.layers.{layer}.mlp.gate.weight"
        header = catalog[tensor]
        if header.values != BLOCK_VALUES:
            raise AssertionError(f"router is not exactly one block: {tensor}")
        router_blocks.append(
            entry_for_block(
                header,
                entry_id=f"l{layer:02d}-router",
                role="mlp.gate.weight",
                layer=layer,
                block_index=0,
                selection_digest="complete-enumeration",
                collision_steps=0,
                stratum_inventory_blocks=1,
                stratum=f"layer-{layer:02d}/mlp.gate.weight",
            )
        )

    rank1_tensors = []
    for tensor, header in sorted(catalog.items()):
        if len(header.shape) != 1:
            continue
        layer, role = normalized_role(tensor)
        byte_range = source_range(header, 0, header.values)
        rank1_tensors.append(
            {
                "id": tensor,
                "layer": layer,
                "role": role,
                "tensor": tensor,
                "shape": list(header.shape),
                "dtype": header.dtype,
                "values": header.values,
                "bytes": 2 * header.values,
                "codec": "lossless raw BF16 exception",
                "shard": header.shard,
                "header_file": header.header_file,
                "absolute_byte_range_in_shard": list(byte_range),
            }
        )

    if len(plte_blocks) != 400:
        raise AssertionError(f"expected 400 new PLTE blocks, found {len(plte_blocks)}")
    if len({(x["tensor"], x["block_index"]) for x in plte_blocks}) != 400:
        raise AssertionError("new PLTE selection contains duplicates")
    if any((x["tensor"], x["block_index"]) in excluded for x in plte_blocks):
        raise AssertionError("new PLTE selection overlaps published evidence")
    if len(router_blocks) != 48 or len(rank1_tensors) != 193:
        raise AssertionError("unexpected complete router or rank-one inventory")

    return {
        "format": "PLTE Qwen3 stratified evaluation manifest v1",
        "checkpoint": expected_checkpoint,
        "strict_ptq": True,
        "selection_reads_weight_payloads": False,
        "selection": {
            "protocol_tag": PROTOCOL_TAG,
            "selection_material": (
                "UTF-8 protocol tag, NUL, immutable Qwen revision, NUL, "
                "base publication commit"
            ),
            "selection_seed_sha256": SELECTION_SEED.hex(),
            "base_publication_commit": BASE_PUBLICATION_COMMIT,
            "algorithm": (
                "SHA-256-derived cyclic selection inside each header-defined stratum; "
                "published block identifiers are skipped without reading their weights"
            ),
            "new_plte_blocks": 400,
            "layer_role_blocks": 336,
            "embedding_blocks": 32,
            "lm_head_blocks": 32,
            "complete_router_blocks_reused": 48,
            "complete_rank1_tensors": 193,
            "published_plte_blocks_excluded": len(excluded),
            "block_values": BLOCK_VALUES,
        },
        "provenance": {
            "header_bundle_sha256": header_bundle_sha256,
            "published_evidence_manifest": evidence_path.name,
            "published_evidence_manifest_sha256": sha256_bytes(evidence_bytes),
            "encoder_sha256": encoder_sha256,
            "independent_decoder_sha256": decoder_sha256,
            "fetcher_sha256": fetcher_sha256,
            "frozen_profile_sha256": profile_sha256,
            "polar_upstream_commit": POLAR_UPSTREAM_COMMIT,
        },
        "inventory": {
            "header_files": 16,
            "checkpoint_tensors": len(catalog),
            "normalized_roles": role_inventory(catalog),
        },
        "plte_blocks": plte_blocks,
        "router_blocks": router_blocks,
        "rank1_tensors": rank1_tensors,
        "coverage_assertions": {
            "all_layers": list(range(48)),
            "plte_layer_roles": list(ATTENTION_ROLES + EXPERT_ROLES),
            "global_plte_roles": ["model.embed_tokens.weight", "lm_head.weight"],
            "router_role": "mlp.gate.weight",
            "rank1_roles": sorted(
                {str(row["role"]) for row in rank1_tensors}
            ),
            "all_15_normalized_roles_accounted_for": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headers",
        type=Path,
        default=PLTE / "qwen_weight_cache" / "headers",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=PLTE / "agent_root_polar_escape_evidence_manifest.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    catalog, header_bundle_sha256 = load_headers(args.headers)
    manifest = build_manifest(catalog, header_bundle_sha256, args.evidence)
    rendered = json.dumps(manifest, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
