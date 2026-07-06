"""Frozen-layer integrity: build & verify autoresearch/frozen/MANIFEST.json.

The MANIFEST records SHA-256 of every frozen file (code + gate tests + job script),
the dataset generation seeds + sidecar hashes, and the tokenizer checkpoint hash.
The driver calls verify() before every scoring run; ANY mismatch means the run is
scored as chance and flagged `tampered` (the loop agent must be unable to move the
goalposts). The driver itself is protected by being OUTSIDE the loop's write
surface (program.md contract + git review), not by self-hashing.

CLI (repo root):
  venv/Scripts/python.exe -m autoresearch.driver.manifest --write   # freeze (human-invoked only)
  venv/Scripts/python.exe -m autoresearch.driver.manifest --check   # verify, exit 1 on mismatch
"""

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FROZEN_DIR = os.path.join(REPO, "autoresearch", "frozen")
MANIFEST_PATH = os.path.join(FROZEN_DIR, "MANIFEST.json")

# Everything under frozen/ except the MANIFEST itself and caches.
def _frozen_files():
    out = []
    for root, dirs, files in os.walk(FROZEN_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if f == "MANIFEST.json" or f.endswith(".pyc"):
                continue
            p = os.path.join(root, f)
            out.append(os.path.relpath(p, REPO).replace(os.sep, "/"))
    return sorted(out)


EXTRA_ARTIFACTS = [
    # (repo-relative path, required) — datasets are regenerable from (code, seed);
    # their sidecar hashes pin the exact bytes; the tokenizer ckpt is the frozen model.
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
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    files = {p: _sha256(os.path.join(REPO, p)) for p in _frozen_files()}
    artifacts = {}
    for p, required in EXTRA_ARTIFACTS:
        full = os.path.join(REPO, p)
        if not os.path.isfile(full):
            if required:
                raise FileNotFoundError(f"required frozen artifact missing: {p}")
            continue
        artifacts[p] = _sha256(full)
    return {
        "version": "colorfield-frozen-v2.1",
        "signed_off": "Merlin 2026-07-06 (border exclusion, multiplicative composite, max-gap age)",
        "dataset_seeds": {"train": 0, "val": 777, "regen": "python -m autoresearch.frozen.datagen"},
        "files": files,
        "artifacts": artifacts,
    }


def verify():
    """Returns (ok: bool, mismatches: list[str]). Missing manifest => not ok."""
    if not os.path.isfile(MANIFEST_PATH):
        return False, ["MANIFEST.json missing"]
    with open(MANIFEST_PATH) as f:
        man = json.load(f)
    mismatches = []
    for section in ("files", "artifacts"):
        for p, want in man.get(section, {}).items():
            full = os.path.join(REPO, p)
            if not os.path.isfile(full):
                mismatches.append(f"missing: {p}")
            elif _sha256(full) != want:
                mismatches.append(f"hash mismatch: {p}")
    current = set(_frozen_files())
    recorded = set(man.get("files", {}))
    for p in sorted(current - recorded):
        mismatches.append(f"unrecorded file in frozen/: {p}")
    return not mismatches, mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.write:
        man = build()
        with open(MANIFEST_PATH, "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        print(f"wrote {MANIFEST_PATH}: {len(man['files'])} files, "
              f"{len(man['artifacts'])} artifacts")
    else:
        ok, mismatches = verify()
        if ok:
            print("MANIFEST OK")
        else:
            print("MANIFEST VIOLATIONS:")
            for m in mismatches:
                print(f"  - {m}")
            sys.exit(1)


if __name__ == "__main__":
    main()
