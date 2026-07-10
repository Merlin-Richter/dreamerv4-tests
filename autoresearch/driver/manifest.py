"""Frozen-layer integrity: build & verify the frozen MANIFEST.json per tier.

Tiers:
  --tier pixel (default): autoresearch/frozen/            -> frozen/MANIFEST.json
  --tier sym:             autoresearch/frozen_sym/ + loop/ -> frozen_sym/MANIFEST.json
    (loop/ is hashed too — program.md declares it NOT agent-editable; the sym
    tier is tokenizer-free so its artifacts are just the dataset sidecars.)

The MANIFEST records SHA-256 of every frozen file (code + gate tests + job
scripts), the dataset generation seeds + sidecar hashes, and (pixel tier) the
tokenizer checkpoint hash. The driver/payload call verify() before every
scoring run; ANY mismatch means the run is void and flagged `tampered` (the
loop agent must be unable to move the goalposts). The driver itself — this
file included — is protected by being OUTSIDE the loop's write surface
(program.md contract + git review of the branch), not by self-hashing: a
manifest check inside an agent-writable repo is FRICTION plus loud accident
detection, and the human review of kept diffs is the actual backstop.

CLI (repo root; stdlib-only — runs under any python3, no venv needed):
  python -m autoresearch.driver.manifest --write --tier sym  # freeze (human-invoked only)
  python -m autoresearch.driver.manifest --check --tier sym  # verify, exit 1 on mismatch
"""

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TIERS = {
    "pixel": {
        "dirs": [os.path.join("autoresearch", "frozen")],
        "manifest": os.path.join("autoresearch", "frozen", "MANIFEST.json"),
        "version": "colorfield-frozen-v2.1",
        "signed_off": "Merlin 2026-07-06 (border exclusion, multiplicative composite, max-gap age)",
        "dataset_seeds": {"train": 0, "val": 777, "regen": "python -m autoresearch.frozen.datagen"},
        "artifacts": [
            # (repo-relative path, required) — datasets are regenerable from (code, seed);
            # sidecar hashes pin the exact bytes; the tokenizer ckpt is the frozen model.
            ("checkpoints/colorfield/tokenizer.pt", True),
            ("data/colorfield/maps.npy", True),
            ("data/colorfield/starts.npy", True),
            ("data/colorfield/actions.npy", True),
            ("data/colorfield/policy_ids.npy", True),
            ("data/colorfield/ep_seeds.npy", True),
            ("data/colorfield_val/maps.npy", True),
            ("data/colorfield_val/starts.npy", True),
            ("data/colorfield_val/actions.npy", True),
            ("data/colorfield_val/policy_ids.npy", True),
            ("data/colorfield_val/ep_seeds.npy", True),
        ],
    },
    "sym": {
        "dirs": [os.path.join("autoresearch", "frozen_sym"),
                 os.path.join("autoresearch", "loop")],
        "manifest": os.path.join("autoresearch", "frozen_sym", "MANIFEST.json"),
        "version": "colorfield-sym-frozen-v2.2",
        "signed_off": "Merlin 2026-07-10 (v2.2 continuous scoring: "
                      "fid*(0.2*ent + 0.8*composite), move/hold fid split, entropy ramp)",
        "dataset_seeds": {"train": 0, "val": 777,
                          "regen": "python -m autoresearch.frozen_sym.datagen"},
        "artifacts": [
            # sym is tokenizer-free; actions.npy is the byte-identity gate the
            # job payload already enforces (procedural determinism verified
            # cross-backend on ferranti 417029 and vast loop-7e41889).
            ("data/colorfield_sym/actions.npy", True),
            ("data/colorfield_sym_val/actions.npy", True),
        ],
    },
}


def _tier_files(tier):
    """Every file under the tier's dirs except MANIFEST.json and caches."""
    out = []
    for d in TIERS[tier]["dirs"]:
        base = os.path.join(REPO, d)
        for root, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in sorted(files):
                if f == "MANIFEST.json" or f.endswith(".pyc"):
                    continue
                p = os.path.join(root, f)
                out.append(os.path.relpath(p, REPO).replace(os.sep, "/"))
    return sorted(out)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(tier="pixel"):
    t = TIERS[tier]
    files = {p: _sha256(os.path.join(REPO, p)) for p in _tier_files(tier)}
    artifacts = {}
    for p, required in t["artifacts"]:
        full = os.path.join(REPO, p)
        if not os.path.isfile(full):
            if required:
                raise FileNotFoundError(f"required frozen artifact missing: {p}")
            continue
        artifacts[p] = _sha256(full)
    return {
        "version": t["version"],
        "signed_off": t["signed_off"],
        "dataset_seeds": t["dataset_seeds"],
        "files": files,
        "artifacts": artifacts,
    }


def verify(tier="pixel", check_artifacts=True):
    """Returns (ok: bool, mismatches: list[str]). Missing manifest => not ok.
    check_artifacts=False skips the dataset/ckpt section (for hosts that don't
    hold the data — the job payload's own sha-gate covers it remotely)."""
    manifest_path = os.path.join(REPO, TIERS[tier]["manifest"])
    if not os.path.isfile(manifest_path):
        return False, ["MANIFEST.json missing"]
    with open(manifest_path) as f:
        man = json.load(f)
    mismatches = []
    sections = ("files", "artifacts") if check_artifacts else ("files",)
    for section in sections:
        for p, want in man.get(section, {}).items():
            full = os.path.join(REPO, p)
            if not os.path.isfile(full):
                mismatches.append(f"missing: {p}")
            elif _sha256(full) != want:
                mismatches.append(f"hash mismatch: {p}")
    current = set(_tier_files(tier))
    recorded = set(man.get("files", {}))
    for p in sorted(current - recorded):
        mismatches.append(f"unrecorded file in frozen dirs: {p}")
    return not mismatches, mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--tier", choices=sorted(TIERS), default="pixel")
    ap.add_argument("--no-artifacts", action="store_true",
                    help="check code files only (host has no datasets)")
    args = ap.parse_args()
    manifest_path = os.path.join(REPO, TIERS[args.tier]["manifest"])
    if args.write:
        man = build(args.tier)
        with open(manifest_path, "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        print(f"wrote {manifest_path}: {len(man['files'])} files, "
              f"{len(man['artifacts'])} artifacts")
    else:
        ok, mismatches = verify(args.tier, check_artifacts=not args.no_artifacts)
        if ok:
            print("MANIFEST OK")
        else:
            print("MANIFEST VIOLATIONS:")
            for m in mismatches:
                print(f"  - {m}")
            sys.exit(1)


if __name__ == "__main__":
    main()
