#!/usr/bin/env python3
"""Run subject-excluded ensemble inference and uncertainty decomposition."""

import os
import sys
import json
import argparse
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from pathlib import Path

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

NUM_CLASSES = 110
N_MODELS = 5

MI_TOLERANCE = 1e-4

REGION_NAMES = [
    "Background", "L-Cerebral-WM", "L-Lateral-Ventricle", "L-Inf-Lat-Ventricle",
    "L-Cerebellum-WM", "L-Cerebellum-Cortex", "L-Thalamus", "L-Caudate", "L-Putamen",
    "L-Pallidum", "3rd-Ventricle", "4th-Ventricle", "Brain-Stem", "L-Hippocampus",
    "L-Amygdala", "CSF", "L-Accumbens", "L-VentralDC", "L-Vessel", "L-Choroid-Plexus",
    "R-Cerebral-WM", "R-Lateral-Ventricle", "R-Inf-Lat-Ventricle", "R-Cerebellum-WM",
    "R-Cerebellum-Cortex", "R-Thalamus", "R-Caudate", "R-Putamen", "R-Pallidum",
    "R-Hippocampus", "R-Amygdala", "R-Accumbens", "R-VentralDC", "R-Vessel",
    "R-Choroid-Plexus", "WM-Hypointensities", "Optic-Chiasm", "CC-Posterior",
    "CC-Mid-Posterior", "CC-Central", "CC-Mid-Anterior", "CC-Anterior",
    "L-bankssts", "L-caudalanteriorcingulate", "L-caudalmiddlefrontal", "L-cuneus",
    "L-entorhinal", "L-fusiform", "L-inferiorparietal", "L-inferiortemporal",
    "L-isthmuscingulate", "L-lateraloccipital", "L-lateralorbitofrontal", "L-lingual",
    "L-medialorbitofrontal", "L-middletemporal", "L-parahippocampal", "L-paracentral",
    "L-parsopercularis", "L-parsorbitalis", "L-parstriangularis", "L-pericalcarine",
    "L-postcentral", "L-posteriorcingulate", "L-precentral", "L-precuneus",
    "L-rostralanteriorcingulate", "L-rostralmiddlefrontal", "L-superiorfrontal",
    "L-superiorparietal", "L-superiortemporal", "L-supramarginal", "L-frontalpole",
    "L-temporalpole", "L-transversetemporal", "L-insula",
    "R-bankssts", "R-caudalanteriorcingulate", "R-caudalmiddlefrontal", "R-cuneus",
    "R-entorhinal", "R-fusiform", "R-inferiorparietal", "R-inferiortemporal",
    "R-isthmuscingulate", "R-lateraloccipital", "R-lateralorbitofrontal", "R-lingual",
    "R-medialorbitofrontal", "R-middletemporal", "R-parahippocampal", "R-paracentral",
    "R-parsopercularis", "R-parsorbitalis", "R-parstriangularis", "R-pericalcarine",
    "R-postcentral", "R-posteriorcingulate", "R-precentral", "R-precuneus",
    "R-rostralanteriorcingulate", "R-rostralmiddlefrontal", "R-superiorfrontal",
    "R-superiorparietal", "R-superiortemporal", "R-supramarginal", "R-frontalpole",
    "R-temporalpole", "R-transversetemporal", "R-insula",
]
assert len(REGION_NAMES) == NUM_CLASSES

TARGET_REGIONS = [
    "L-Hippocampus", "R-Hippocampus", "L-Amygdala", "R-Amygdala",
    "L-Thalamus", "R-Thalamus", "L-Caudate", "R-Caudate",
    "L-Putamen", "R-Putamen", "L-Pallidum", "R-Pallidum",
    "L-Cerebral-WM", "R-Cerebral-WM", "Brain-Stem",
]
TARGET_IDX = [REGION_NAMES.index(r) for r in TARGET_REGIONS]


def build_group_map(splits_path):
    """Build the subject-to-group map and check group disjointness."""
    with open(splits_path) as f:
        splits = json.load(f)
    if len(splits) != N_MODELS:
        sys.exit(f"ERROR: expected {N_MODELS} entries in {splits_path}, got {len(splits)}")

    groups = [set(s["train"]) for s in splits]

    for a in range(N_MODELS):
        for b in range(a + 1, N_MODELS):
            overlap = groups[a] & groups[b]
            if overlap:
                sys.exit(
                    f"ERROR: groups {a} and {b} share {len(overlap)} subjects.\n"
                    f"       This splits file is NOT the disjoint-fifths design.\n"
                    f"       Check that --splits points to splits_final_disjoint.json."
                )

    gmap = {}
    for i, g in enumerate(groups):
        for sid in g:
            gmap[sid] = i

    sizes = [len(g) for g in groups]
    print(f"Group map: {len(gmap)} subjects across {N_MODELS} disjoint groups {sizes}")
    return gmap


def predict_subset_with_decomposition(predictor, data, model_indices):
    """Ensemble the selected models and decompose their uncertainty."""
    n_threads = torch.get_num_threads()
    torch.set_num_threads(8)

    softmax_sum = None
    member_entropy_sum = None

    for i in model_indices:
        params = predictor.list_of_parameters[i]
        if not isinstance(predictor.network, torch.nn.Module):
            predictor.network._orig_mod.load_state_dict(params)
        else:
            try:
                predictor.network.load_state_dict(params)
            except Exception:
                predictor.network._orig_mod.load_state_dict(params)

        logits = predictor.predict_sliding_window_return_logits(data)
        softmax = torch.softmax(logits.float(), dim=0).cpu().numpy()
        del logits

        h_m = np.zeros(softmax.shape[1:], dtype=np.float32)
        for c in range(softmax.shape[0]):
            pc = softmax[c]
            np.clip(pc, 1e-7, 1.0, out=pc)
            h_m -= pc * np.log(pc)

        if softmax_sum is None:
            softmax_sum = softmax
            member_entropy_sum = h_m
        else:
            softmax_sum += softmax
            member_entropy_sum += h_m
        del softmax, h_m

    n = len(model_indices)
    mean_softmax = softmax_sum / n
    del softmax_sum

    total_entropy = np.zeros(mean_softmax.shape[1:], dtype=np.float32)
    for c in range(mean_softmax.shape[0]):
        pc = mean_softmax[c]
        np.clip(pc, 1e-7, 1.0, out=pc)
        total_entropy -= pc * np.log(pc)

    mean_member_entropy = member_entropy_sum / n
    del member_entropy_sum

    mi_raw = total_entropy - mean_member_entropy
    mi_min = float(mi_raw.min())
    if mi_min < -MI_TOLERANCE:
        raise ValueError(
            f"mutual information reached {mi_min:.3e}, below tolerance "
            f"-{MI_TOLERANCE:.0e}. This violates Jensen's inequality and "
            f"indicates a computation error, not numerical noise. "
            f"Check the member accumulation and n = len(model_indices)."
        )
    mutual_information = np.clip(mi_raw, 0.0, None)
    del mi_raw

    msp_uncertainty = 1.0 - mean_softmax.max(axis=0)

    torch.set_num_threads(n_threads)
    return (mean_softmax, total_entropy, mean_member_entropy,
            mutual_information, msp_uncertainty)


def revert_preprocessing(volume, properties, plans_manager, is_seg=False):
    """Return a predicted volume to the original image space."""
    from scipy.ndimage import zoom

    target_shape = properties['shape_after_cropping_and_before_resampling']

    if volume.ndim == 3:
        if volume.shape != tuple(target_shape):
            zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
            order = 0 if is_seg else 1
            volume = zoom(volume, zoom_factors, order=order)

        original_shape = properties['shape_before_cropping']
        dtype = volume.dtype if is_seg else np.float32
        reverted = np.zeros(original_shape, dtype=dtype)
        slicer = bounding_box_to_slice(properties['bbox_used_for_cropping'])
        reverted[slicer] = volume

    elif volume.ndim == 4:
        C = volume.shape[0]
        if volume.shape[1:] != tuple(target_shape):
            zoom_factors = [1] + [t / s for t, s in zip(target_shape, volume.shape[1:])]
            volume = zoom(volume, zoom_factors, order=1)
        original_shape = properties['shape_before_cropping']
        reverted = np.zeros((C, *original_shape), dtype=np.float32)
        slicer = bounding_box_to_slice(properties['bbox_used_for_cropping'])
        reverted[(slice(None),) + slicer] = volume

    tb = plans_manager.transpose_backward
    if volume.ndim == 3:
        reverted = reverted.transpose(tb)
    else:
        reverted = reverted.transpose([0] + [t + 1 for t in tb])
    return reverted


def to_native(vol, properties, plans_manager, is_seg=False):
    """Return a volume to the native orientation."""
    out = revert_preprocessing(vol, properties, plans_manager, is_seg=is_seg)
    out = np.transpose(out, (2, 1, 0))
    return out.astype(np.uint8 if is_seg else np.float32)


def region_stats(label_vol, maps):
    """Summarise each map inside the reference regions."""
    row = {}
    for idx, name in enumerate(REGION_NAMES):
        mask = (label_vol == idx)
        n = int(mask.sum())
        row[f"{name}_n_voxels"] = n
        for mname, m in maps.items():
            row[f"{name}_{mname}_mean"] = float(m[mask].mean()) if n > 0 else np.nan
    return row


def prospective_stats(seg_vol, maps):
    """Summarise uncertainty inside predicted regions."""
    row = {}

    tgt = np.isin(seg_vol, TARGET_IDX)
    n_tgt = int(tgt.sum())
    row["predtgt_n_voxels"] = n_tgt
    for mname, m in maps.items():
        row[f"predtgt_{mname}_mean"] = float(m[tgt].mean()) if n_tgt > 0 else np.nan

    fg = seg_vol > 0
    n_fg = int(fg.sum())
    row["predfg_n_voxels"] = n_fg
    for mname, m in maps.items():
        row[f"predfg_{mname}_mean"] = float(m[fg].mean()) if n_fg > 0 else np.nan
    for idx, name in enumerate(REGION_NAMES):
        if idx == 0:
            continue
        mask = (seg_vol == idx)
        n = int(mask.sum())
        row[f"{name}_pred_n_voxels"] = n
        for mname, m in maps.items():
            row[f"{name}_pred_{mname}_mean"] = float(m[mask].mean()) if n > 0 else np.nan
    return row


def region_dice(label_vol, seg_vol):
    """Calculate per-region Dice against the reference."""
    row = {}
    for idx, name in enumerate(REGION_NAMES):
        ref = (label_vol == idx)
        pred = (seg_vol == idx)
        nref, npred = int(ref.sum()), int(pred.sum())
        if nref == 0 and npred == 0:
            row[f"{name}_dice"] = np.nan
        else:
            inter = int(np.logical_and(ref, pred).sum())
            row[f"{name}_dice"] = 2.0 * inter / (nref + npred)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='imagesTr folder')
    ap.add_argument('--labels', required=True, help='labelsTr folder (FreeSurfer reference)')
    ap.add_argument('--splits', required=True,
                    help='path to splits_final_disjoint.json')
    ap.add_argument('--model_dir', required=True,
                    help='.../nnUNetTrainer_250epochs_disjoint__nnUNetPlans__2d')
    ap.add_argument('--output', required=True)
    ap.add_argument('--checkpoint', default='checkpoint_final.pth')
    ap.add_argument('--save_maps', action='store_true',
                    help='also write the total-entropy NIfTI per subject (~36 GB for 532)')
    ap.add_argument('--test', action='store_true',
                    help='process one subject from each disjoint group')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 68)
    print("Disjoint-fifths leakage-free inference with uncertainty decomposition")
    print("=" * 68)
    gmap = build_group_map(args.splits)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_dir,
        use_folds=tuple(range(N_MODELS)),
        checkpoint_name=args.checkpoint,
    )
    n_loaded = len(predictor.list_of_parameters)
    if n_loaded != N_MODELS:
        sys.exit(f"ERROR: loaded {n_loaded} models, expected {N_MODELS}")
    print(f"Loaded {n_loaded} model weight sets, TTA mirroring OFF\n")

    input_files = sorted(Path(args.input).glob('*_0000.nii.gz'))

    if args.test:
        picked, seen = [], set()
        for f in input_files:
            sid = f.name.replace('_0000.nii.gz', '')
            g = gmap.get(sid)
            if g is not None and g not in seen:
                seen.add(g)
                picked.append(f)
            if len(seen) == N_MODELS:
                break
        if len(seen) < N_MODELS:
            print(f"WARNING: --test covered only groups {sorted(seen)}")
        input_files = picked
        print(f"TEST MODE: {len(input_files)} subjects, one per group "
              f"{sorted(seen)}\n")
    else:
        print(f"Processing {len(input_files)} subjects\n")

    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    preprocessor = DefaultPreprocessor()
    rw = SimpleITKIO()

    out_csv = os.path.join(args.output,
                           "region_entropy_decomposed_reference_masked.csv")
    partial_csv = out_csv + ".partial"

    done = set()
    if os.path.exists(partial_csv):
        try:
            prev = pd.read_csv(partial_csv)
            done = set(prev["subject_id"].astype(str))
            print(f"RESUMING: {len(done)} subjects already in {partial_csv}\n")
        except Exception as e:
            print(f"WARNING: could not read {partial_csv} ({e}); starting fresh")

    rows = []
    missing_group = []
    n_skipped_shape = 0
    n_skipped_affine = 0
    n_skipped_nolabel = 0

    for k, f in enumerate(input_files):
        sid = f.name.replace('_0000.nii.gz', '')

        if sid in done:
            continue

        if sid not in gmap:
            missing_group.append(sid)
            print(f"[{k+1}/{len(input_files)}] {sid}  SKIPPED - not in any group")
            continue

        own = gmap[sid]
        use = [i for i in range(N_MODELS) if i != own]

        lab_path = os.path.join(args.labels, f"{sid}.nii.gz")
        if not os.path.exists(lab_path):
            print(f"[{k+1}/{len(input_files)}] {sid}  SKIPPED - no reference label")
            n_skipped_nolabel += 1
            continue

        print(f"[{k+1}/{len(input_files)}] {sid}  group={own}  models={use}",
              end=" ", flush=True)

        image, properties = rw.read_images([str(f)])
        data, _ = preprocessor.run_case_npy(
            image, None, properties,
            predictor.plans_manager, predictor.configuration_manager,
            predictor.dataset_json,
        )
        data = torch.from_numpy(data).to(dtype=torch.float32, device=predictor.device)

        mean_softmax, total, mme, mi, msp = predict_subset_with_decomposition(
            predictor, data, use)
        del data

        seg = np.argmax(mean_softmax, axis=0).astype(np.uint8)
        del mean_softmax

        total = to_native(total, properties, predictor.plans_manager)
        mme = to_native(mme, properties, predictor.plans_manager)
        mi = to_native(mi, properties, predictor.plans_manager)
        msp = to_native(msp, properties, predictor.plans_manager)
        seg = to_native(seg, properties, predictor.plans_manager, is_seg=True)

        lab_img = nib.load(lab_path)
        lab = np.asarray(lab_img.dataobj).astype(np.int16)
        if lab.shape != total.shape:
            print(f"SKIPPED - shape mismatch {total.shape} vs {lab.shape}")
            n_skipped_shape += 1
            continue

        src_affine = nib.load(str(f)).affine
        if not np.allclose(src_affine, lab_img.affine, atol=1e-3):
            print(f"SKIPPED - affine mismatch between image and reference label")
            n_skipped_affine += 1
            continue

        row = {"subject_id": sid, "group": own, "models_used": "|".join(map(str, use))}
        maps = {
            "entropy": total,
            "mean_member_entropy": mme,
            "mutual_information": mi,
            "msp_unc": msp,
        }
        row.update(region_stats(lab, maps))
        row.update(region_dice(lab, seg))
        row.update(prospective_stats(seg, maps))
        rows.append(row)

        hdr = not os.path.exists(partial_csv)
        pd.DataFrame([row]).to_csv(partial_csv, mode='a', header=hdr, index=False)

        if args.save_maps:
            ref = nib.load(str(f))
            nib.save(nib.Nifti1Image(total, ref.affine, ref.header),
                     os.path.join(args.output, f"{sid}_entropy.nii.gz"))
            nib.save(nib.Nifti1Image(seg, ref.affine, ref.header),
                     os.path.join(args.output, f"{sid}_seg.nii.gz"))

        _h = np.nanmean([row[f'{r}_entropy_mean'] for r in REGION_NAMES[1:]])
        _d = np.nanmean([row[f'{r}_dice'] for r in REGION_NAMES[1:]])
        print(f"H={_h:.4f} Dice={_d:.4f}")

        del total, mme, mi, msp, seg, lab
        torch.cuda.empty_cache()

    df = pd.read_csv(partial_csv) if os.path.exists(partial_csv) \
        else pd.DataFrame(rows)

    n_expected = len(gmap)
    n_got = len(df)
    print("\n" + "=" * 68)
    print(f"Completeness: {n_got} / {n_expected} expected subjects")
    if n_skipped_nolabel:
        print(f"  skipped, no reference label : {n_skipped_nolabel}")
    if n_skipped_shape:
        print(f"  skipped, shape mismatch     : {n_skipped_shape}")
    if n_skipped_affine:
        print(f"  skipped, affine mismatch    : {n_skipped_affine}")
    if missing_group:
        print(f"  skipped, no group           : {len(missing_group)}"
              f"  {missing_group[:10]}")

    if args.test:
        print("TEST MODE - completeness not enforced.")
        df.to_csv(out_csv, index=False)
        print(f"Wrote {n_got} subjects -> {out_csv}")
    elif n_got == n_expected:
        df.to_csv(out_csv, index=False)
        os.remove(partial_csv)
        print(f"COMPLETE. Wrote {n_got} subjects -> {out_csv}")
    else:
        print(f"\nINCOMPLETE - {n_expected - n_got} subjects missing.")
        print(f"Results left in {partial_csv}, NOT promoted to {out_csv}.")
        print("Resolve the skips above and re-run; done subjects are skipped.")
        sys.exit(1)

    tgt = ["L-Hippocampus", "R-Hippocampus", "L-Amygdala", "R-Amygdala",
           "L-Thalamus", "R-Thalamus", "L-Caudate", "R-Caudate",
           "L-Putamen", "R-Putamen", "L-Pallidum", "R-Pallidum",
           "L-Cerebral-WM", "R-Cerebral-WM", "Brain-Stem"]
    print("\nHeadline numbers (compare against the current thesis):")
    all_d = [f"{r}_dice" for r in REGION_NAMES[1:] if f"{r}_dice" in df.columns]
    tgt_d = [f"{r}_dice" for r in tgt if f"{r}_dice" in df.columns]
    print(f"  mean Dice, 109 regions : {df[all_d].stack().mean():.4f}   "
          f"(thesis out-of-fold single model: 0.46)")
    print(f"  mean Dice, 15 targets  : {df[tgt_d].mean(axis=1).mean():.4f}   "
          f"(thesis matched single model: 0.594)")
    for r in ["L-Hippocampus", "Brain-Stem", "L-Caudate"]:
        if f"{r}_dice" in df.columns:
            print(f"    {r:<16}: {df[f'{r}_dice'].mean():.4f}")
    if missing_group:
        print(f"WARNING: {len(missing_group)} subjects had no group: {missing_group[:10]}")

    print("\n" + "=" * 68)
    print("Group-effect check (ensemble composition varies by group)")
    print("=" * 68)
    targets = ["L-Hippocampus", "R-Hippocampus", "L-Thalamus", "R-Thalamus",
               "Brain-Stem", "L-Cerebral-WM", "R-Cerebral-WM"]
    cols = [f"{t}_entropy_mean" for t in targets if f"{t}_entropy_mean" in df.columns]
    dcols = [f"{t}_dice" for t in targets if f"{t}_dice" in df.columns]
    if cols:
        df["_target_mean_entropy"] = df[cols].mean(axis=1)
        df["_target_mean_dice"] = df[dcols].mean(axis=1)
        summary = df.groupby("group")[["_target_mean_entropy", "_target_mean_dice"]].agg(
            ["count", "mean", "std"])
        print(summary.to_string())
        for q in ["_target_mean_entropy", "_target_mean_dice"]:
            m = df.groupby("group")[q].mean()
            sd = df.groupby("group")[q].std().mean()
            print(f"\n{q}: max-min across groups = {m.max()-m.min():.4f}"
                  f"  (mean within-group SD = {sd:.4f})")
        print("\nA between-group spread comparable to or larger than the within-group")
        print("SD suggests one model is materially weaker than the others. Because")
        print("each group is served by a different four-model subset, that would")
        print("propagate to the four groups it predicts and become a confound for")
        print("RQ2/RQ3. If it appears, add group as a covariate and say so.")
        print("\nRegardless of the outcome above, check covariate balance across")
        print("the five groups (sex, pathology, operation side) after merging with")
        print("the clinical metadata. The groups were assigned at random, but with")
        print("~106 subjects each an imbalance is possible, and group is now")
        print("confounded with ensemble composition. Include group as a nuisance")
        print("variable in the regression, or report it as a sensitivity analysis.")


if __name__ == '__main__':
    main()
