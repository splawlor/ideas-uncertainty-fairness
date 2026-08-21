#!/usr/bin/env python3
"""Create the disjoint-fifths split."""

import argparse
import json
import os
import sys
from itertools import combinations

def load_source(source_path):
    if not os.path.isfile(source_path):
        sys.exit(f"ERROR: {source_path} not found")
    with open(source_path) as f:
        splits = json.load(f)
    if len(splits) != 5:
        sys.exit(f"ERROR: expected 5 folds, found {len(splits)}")
    return splits


def describe_source(splits):
    print("=== SOURCE splits_final.json ===")
    total = set()
    for i, f in enumerate(splits):
        print(f"  fold {i}: train={len(f['train']):4d}  val={len(f['val']):4d}")
        total |= set(f["train"]) | set(f["val"])
    print(f"  distinct subjects across all folds: {len(total)}")

    print("\n  training-set overlap between model pairs (Problem B):")
    for a, b in combinations(range(5), 2):
        shared = len(set(splits[a]["train"]) & set(splits[b]["train"]))
        print(f"    model {a} vs {b}: {shared} shared training subjects")
    return total


def build_disjoint(splits):
    groups = [list(f["val"]) for f in splits]

    for a, b in combinations(range(5), 2):
        inter = set(groups[a]) & set(groups[b])
        if inter:
            sys.exit(f"ERROR: groups {a} and {b} overlap ({len(inter)} subjects)")

    new = []
    for i in range(5):
        new.append({
            "train": sorted(groups[i]),
            "val": sorted(groups[(i + 1) % 5]),
        })
    return new, groups


def verify(new, groups, all_subjects):
    print("\n=== NEW disjoint-fifths splits ===")
    for i, f in enumerate(new):
        print(f"  model {i}: train={len(f['train']):4d} (group {i})"
              f"  val={len(f['val']):4d} (group {(i+1)%5})")

    print("\n  training-set overlap between model pairs (must all be 0):")
    bad = False
    for a, b in combinations(range(5), 2):
        shared = len(set(new[a]["train"]) & set(new[b]["train"]))
        flag = "" if shared == 0 else "   <-- FAIL"
        if shared:
            bad = True
        print(f"    model {a} vs {b}: {shared}{flag}")

    print("\n  train/val disjoint within each model (must all be 0):")
    for i, f in enumerate(new):
        shared = len(set(f["train"]) & set(f["val"]))
        flag = "" if shared == 0 else "   <-- FAIL"
        if shared:
            bad = True
        print(f"    model {i}: {shared}{flag}")

    covered = set()
    for f in new:
        covered |= set(f["train"])
    print(f"\n  subjects covered by union of training sets: {len(covered)}"
          f" / {len(all_subjects)}")
    if covered != set(all_subjects):
        bad = True
        print("    <-- FAIL: training sets do not partition the cohort")

    print("\n  leakage-free ensemble membership per group:")
    for i in range(5):
        clean = [j for j in range(5) if groups[i][0] not in new[j]["train"]]
        clean = [j for j in range(5) if not (set(groups[i]) & set(new[j]["train"]))]
        print(f"    group {i} ({len(groups[i])} subjects): "
              f"predicted by models {clean}")
        if len(clean) != 4 or i in clean:
            bad = True
            print("      <-- FAIL: expected exactly the four models other than "
                  f"model {i}")

    return not bad


def main():
    ap = argparse.ArgumentParser()
    default_root = os.environ.get("nnUNet_preprocessed")
    default_source = (
        os.path.join(default_root, "Dataset001_IDEAS", "splits_final.json")
        if default_root else None
    )
    ap.add_argument(
        "--source",
        default=default_source,
        help=(
            "path to splits_final.json; defaults to "
            "$nnUNet_preprocessed/Dataset001_IDEAS/splits_final.json"
        ),
    )
    ap.add_argument("--write", action="store_true",
                    help="write splits_final_disjoint.json")
    ap.add_argument("--out", default=None,
                    help="write somewhere other than the default target")
    args = ap.parse_args()

    if not args.source:
        ap.error("--source is required when nnUNet_preprocessed is not set")

    source_path = os.path.abspath(args.source)
    splits = load_source(source_path)
    all_subjects = describe_source(splits)
    new, groups = build_disjoint(splits)
    ok = verify(new, groups, all_subjects)

    if not ok:
        sys.exit("\nVERIFICATION FAILED - nothing written")

    print("\nAll checks passed.")

    out_path = args.out or os.path.join(
        os.path.dirname(source_path), "splits_final_disjoint.json"
    )

    if not args.write:
        print(f"\nDRY RUN - nothing written. Re-run with --write to create\n"
              f"  {out_path}")
        return

    with open(out_path, "w") as f:
        json.dump(new, f, indent=2)
    print(f"\nWrote disjoint splits -> {out_path}")
    print(f"Source split unchanged: {source_path}")
    print("\nTrain with: nnUNetv2_train 1 2d <fold> -tr nnUNetTrainer_250epochs_disjoint")
    print("That trainer reads splits_final_disjoint.json via its do_split() override,")
    print("and writes to its own results directory.")


if __name__ == "__main__":
    main()
