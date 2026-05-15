import numpy as np

from eepynet.data.dataset import SleepEDFChunkDataset, compute_class_counts
from eepynet.data.splits import generate_subject_splits
from eepynet.utils import save_json


def test_generate_subject_splits_has_no_subject_overlap():
    records = [
        {"subject_id": f"S{i:02d}", "record_id": f"S{i:02d}1", "num_epochs": 10}
        for i in range(10)
    ]
    manifest = generate_subject_splits(records, 0.7, 0.2, 0.1, seed=7)

    subject_sets = [
        set(manifest["splits"][split]["subject_ids"])
        for split in ("train", "val", "test")
    ]
    assert subject_sets[0].isdisjoint(subject_sets[1])
    assert subject_sets[0].isdisjoint(subject_sets[2])
    assert subject_sets[1].isdisjoint(subject_sets[2])


def test_chunk_dataset_pads_tail_and_masks_invalid_labels(tmp_path):
    processed_dir = tmp_path / "processed"
    record_dir = processed_dir / "R001"
    record_dir.mkdir(parents=True)
    x = np.zeros((2, 5, 3000), dtype=np.float32)
    y = np.array([0, 1, 2, -1, 4], dtype=np.int64)
    np.save(record_dir / "x.npy", x)
    np.save(record_dir / "y.npy", y)
    save_json(
        {
            "record_id": "R001",
            "subject_id": "S001",
            "sample_rate": 100,
            "num_epochs": 5,
        },
        record_dir / "meta.json",
    )
    manifest = {
        "splits": {
            "train": {"subject_ids": ["S001"], "record_ids": ["R001"]},
            "val": {"subject_ids": [], "record_ids": []},
            "test": {"subject_ids": [], "record_ids": []},
        }
    }

    dataset = SleepEDFChunkDataset(processed_dir, manifest, "train", epochs_per_chunk=8, stride=8)
    item = dataset[0]

    assert item["x"].shape == (2, 8, 3000)
    assert item["y"].tolist()[:5] == [0, 1, 2, 0, 4]
    assert item["mask"].tolist() == [True, True, True, False, True, False, False, False]
    assert compute_class_counts(processed_dir, manifest, "train").tolist() == [1, 1, 1, 0, 1]
