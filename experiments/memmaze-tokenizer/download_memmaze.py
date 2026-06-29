#!/usr/bin/env python3
"""Download Memory Maze 9x9 offline-dataset zip shards from Google Drive (public) via gdown.

The 9x9 set (drive folder 1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB -> memory-maze-9x9) is packaged as 11
public SINGLE files: eval.zip + train-part0..9.zip. We pull whole single files (NOT a Drive folder),
so gdown's 50-files-per-folder limit does not apply. Each train-partN.zip is ~1/10 of the 29k-trajectory
train set (~10 GB). Quota throttling on a popular public file is possible -- if a part fails with a
quota/permission error we keep whatever parts succeeded (train on what we got).

Run on the cluster (data must live on the cluster). With --unzip each .zip is extracted in place.
    python -u experiments/memmaze-tokenizer/download_memmaze.py \
        --parts train-part0 --out-dir data/memmaze9x9_raw --unzip
"""
import argparse
import sys
import zipfile
from pathlib import Path

# File IDs read off the public Drive folder (memory-maze-9x9), 2026-06-29.
PARTS = {
    "eval": "18YPeLnu_TkVx7r4mZjeQ9gSMPEI-jtiP",
    "train-part0": "1EdWafDjZG3VUo7CXTERx7fTx4IqWWNvC",
    "train-part1": "1Kyn8seqBiU8drJ8uyS2RSgeefLatUCzi",
    "train-part2": "1FaxOJJu6hbPjkab5dtCu-QWxMVDEcj5T",
    "train-part3": "1S6G-knZUG2V0Jhobseb_2-_V5n2jhVMS",
    "train-part4": "1br0-EZfh4aTY5E66_m-PXR9i0v1IA1-8",
    "train-part5": "17ZGVQ8fGVv9FlEQiLy-j_M9r4NlABY3i",
    "train-part6": "10vPLCDjv4AC35TkdwAkgUrrAvAhIQJ9v",
    "train-part7": "1vebIvddC4UG78YSpbGTJlr-Hl1ScZeGV",
    "train-part8": "1KmVoAofGWnwBJ0EqClYqWNBzENMA8riE",
    "train-part9": "1N4eiw0DV-HrxWSkhRDmBV6skGTiOUz5B",
}


def resolve_parts(requested):
    out = []
    for p in requested:
        if p == "all":
            return sorted(PARTS)
        if p == "all-train":
            return [k for k in sorted(PARTS) if k.startswith("train")]
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["train-part0"],
                    help="Shard names (train-part0..9, eval) or 'all' / 'all-train'.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--unzip", action="store_true", help="Extract each .zip after download.")
    args = ap.parse_args()

    import gdown

    parts = resolve_parts(args.parts)
    bad = [p for p in parts if p not in PARTS]
    if bad:
        sys.exit(f"unknown parts {bad}; valid: {sorted(PARTS)} (or all / all-train)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for name in parts:
        dest = args.out_dir / f"{name}.zip"
        url = f"https://drive.google.com/uc?id={PARTS[name]}"
        print(f"=== {name} -> {dest} ===", flush=True)
        try:
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  already present ({dest.stat().st_size / 1e9:.2f} GB), skipping download", flush=True)
            else:
                got = gdown.download(url, str(dest), quiet=False)
                if not got:
                    raise RuntimeError("gdown returned None (download quota exceeded / not public?)")
            if args.unzip:
                exdir = args.out_dir / name
                exdir.mkdir(exist_ok=True)
                print(f"  unzip -> {exdir}", flush=True)
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(exdir)
            ok.append(name)
        except Exception as e:  # noqa: BLE001 - report and continue to next shard
            print(f"  FAILED {name}: {e}", flush=True)
            failed.append(name)

    print(f"\nDONE ok={ok} failed={failed}", flush=True)
    if not ok:
        sys.exit("no parts downloaded")


if __name__ == "__main__":
    main()
