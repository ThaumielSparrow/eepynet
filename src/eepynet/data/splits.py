from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from eepynet.config import load_config
from eepynet.utils import load_json, save_json


def load_processed_records(processed_dir: str | Path) -> list[dict]:
    records: list[dict] = []
    for meta_path in sorted(Path(processed_dir).glob("*/meta.json")):
        meta = load_json(meta_path)
        if int(meta.get("num_epochs", 0)) > 0:
            records.append(meta)
    if not records:
        raise FileNotFoundError(f"No processed records found under {processed_dir}")
    return records


def generate_subject_splits(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    records_by_subject: dict[str, list[str]] = defaultdict(list)
    for record in records:
        records_by_subject[str(record["subject_id"])].append(str(record["record_id"]))

    subjects = np.array(sorted(records_by_subject))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(round(n_subjects * train_ratio))
    n_val = int(round(n_subjects * val_ratio))
    n_train = min(max(n_train, 1), n_subjects - 2)
    n_val = min(max(n_val, 1), n_subjects - n_train - 1)

    train_subjects = sorted(subjects[:n_train].tolist())
    val_subjects = sorted(subjects[n_train : n_train + n_val].tolist())
    test_subjects = sorted(subjects[n_train + n_val :].tolist())

    def records_for(subject_ids: list[str]) -> list[str]:
        out: list[str] = []
        for subject_id in subject_ids:
            out.extend(sorted(records_by_subject[subject_id]))
        return sorted(out)

    return {
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "splits": {
            "train": {
                "subject_ids": train_subjects,
                "record_ids": records_for(train_subjects),
            },
            "val": {
                "subject_ids": val_subjects,
                "record_ids": records_for(val_subjects),
            },
            "test": {
                "subject_ids": test_subjects,
                "record_ids": records_for(test_subjects),
            },
        },
    }


def write_split_manifest(config: dict) -> dict:
    split_cfg = config["splits"]
    records = load_processed_records(config["paths"]["processed_dir"])
    manifest = generate_subject_splits(
        records=records,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
    )
    save_json(manifest, config["paths"]["split_path"])
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create subject-level Sleep-EDF splits.")
    parser.add_argument("--config", default="configs/eepynet.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    manifest = write_split_manifest(config)
    for split, payload in manifest["splits"].items():
        print(
            f"{split}: {len(payload['subject_ids'])} subjects, "
            f"{len(payload['record_ids'])} records"
        )


if __name__ == "__main__":
    main()
