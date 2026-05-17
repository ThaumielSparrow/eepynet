import numpy as np

from eepynet.constants import STAGE_TO_ID, normalize_stage_label
from eepynet.data.preprocess import annotations_to_epoch_labels, trim_excess_wake


def test_normalize_stage_label_maps_sleep_edf_descriptions():
    assert normalize_stage_label("Sleep stage W") == STAGE_TO_ID["W"]
    assert normalize_stage_label("Sleep stage 1") == STAGE_TO_ID["N1"]
    assert normalize_stage_label("Sleep stage 2") == STAGE_TO_ID["N2"]
    assert normalize_stage_label("Sleep stage 3") == STAGE_TO_ID["N3"]
    assert normalize_stage_label("Sleep stage 4") == STAGE_TO_ID["N3"]
    assert normalize_stage_label("Sleep stage R") == STAGE_TO_ID["REM"]
    assert normalize_stage_label("Movement time") is None
    assert normalize_stage_label("Sleep stage ?") is None


def test_annotations_to_epoch_labels_uses_max_overlap():
    labels = annotations_to_epoch_labels(
        onsets=[0, 60, 90],
        durations=[60, 30, 30],
        descriptions=["Sleep stage W", "Sleep stage 1", "Sleep stage R"],
        num_epochs=4,
        epoch_seconds=30,
    )

    assert labels.tolist() == [
        STAGE_TO_ID["W"],
        STAGE_TO_ID["W"],
        STAGE_TO_ID["N1"],
        STAGE_TO_ID["REM"],
    ]


def test_trim_excess_wake_keeps_boundary_wake_epochs():
    x = np.zeros((2, 8, 3000), dtype=np.float32)
    y = np.array([0, 0, 0, 2, 2, 0, 0, 0], dtype=np.int64)

    trimmed_x, trimmed_y, meta = trim_excess_wake(x, y, keep_wake_epochs=1)

    assert trimmed_x.shape == (2, 4, 3000)
    assert trimmed_y.tolist() == [0, 2, 2, 0]
    assert meta == {"trim_start_epoch": 2, "trim_end_epoch": 6}
