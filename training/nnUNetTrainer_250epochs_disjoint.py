"""250-epoch nnU-Net trainer using the disjoint-fifths split."""

import os
import json

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

SPLITS_FILENAME = "splits_final_disjoint.json"


class nnUNetTrainer_250epochs_disjoint(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 250
        self.save_every = 25

    def do_split(self):
        """Read and validate splits_final_disjoint.json."""
        splits_file = os.path.join(self.preprocessed_dataset_folder_base,
                                   SPLITS_FILENAME)
        if not os.path.isfile(splits_file):
            raise FileNotFoundError(
                f"{SPLITS_FILENAME} not found in "
                f"{self.preprocessed_dataset_folder_base}. "
                "Run make_disjoint_splits.py first."
            )

        with open(splits_file) as f:
            splits = json.load(f)

        if len(splits) != 5:
            raise ValueError(f"expected 5 entries in {SPLITS_FILENAME}, "
                             f"got {len(splits)}")

        train_sets = [set(s["train"]) for s in splits]
        for a in range(5):
            for b in range(a + 1, 5):
                overlap = train_sets[a] & train_sets[b]
                if overlap:
                    raise ValueError(
                        f"training sets {a} and {b} share {len(overlap)} "
                        f"subjects -- this is not a disjoint design. "
                        f"Did make_disjoint_splits.py write correctly?"
                    )

        tr_keys = sorted(splits[self.fold]["train"])
        val_keys = sorted(splits[self.fold]["val"])

        if set(tr_keys) & set(val_keys):
            raise ValueError(f"fold {self.fold}: train and val overlap")

        self.print_to_log_file(
            f"Disjoint design, fold {self.fold}: "
            f"training on group {self.fold} ({len(tr_keys)} subjects), "
            f"validating on group {(self.fold + 1) % 5} ({len(val_keys)} subjects)"
        )
        return tr_keys, val_keys
