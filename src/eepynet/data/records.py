from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EDFPair:
    record_id: str
    subject_id: str
    study: str
    psg_path: str
    hypnogram_path: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def record_key(path: str | Path) -> str:
    stem = Path(path).stem
    return stem.split("-")[0][:6]


def subject_key(record_id: str) -> str:
    return record_id[:5]


def _filter_studies(paths: Iterable[Path], include_studies: list[str] | None) -> list[Path]:
    selected = []
    allowed = set(include_studies) if include_studies else None
    for path in paths:
        if allowed is None or path.parent.name in allowed:
            selected.append(path)
    return selected


def discover_edf_pairs(
    data_root: str | Path,
    include_studies: list[str] | None = None,
) -> list[EDFPair]:
    data_root = Path(data_root)
    psg_files = _filter_studies(data_root.rglob("*-PSG.edf"), include_studies)
    hypnogram_files = _filter_studies(data_root.rglob("*-Hypnogram.edf"), include_studies)

    hypnograms_by_record: dict[str, Path] = {}
    for hypnogram_path in sorted(hypnogram_files):
        key = record_key(hypnogram_path)
        hypnograms_by_record.setdefault(key, hypnogram_path)

    pairs: list[EDFPair] = []
    for psg_path in sorted(psg_files):
        record_id = record_key(psg_path)
        hypnogram_path = hypnograms_by_record.get(record_id)
        if hypnogram_path is None:
            continue
        pairs.append(
            EDFPair(
                record_id=record_id,
                subject_id=subject_key(record_id),
                study=psg_path.parent.name,
                psg_path=str(psg_path),
                hypnogram_path=str(hypnogram_path),
            )
        )

    if not pairs:
        raise FileNotFoundError(f"No paired PSG/Hypnogram EDF files found under {data_root}")
    return pairs
