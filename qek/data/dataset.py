from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pulser as pl


@dataclass
class ProcessedData:
    """
    Data class that contains:
    - a sequence (i.e. a register + a pulse)
    - the state dictionary, converted to int from a numpy.int64 array for serialization purposes
    - the target
    """

    sequence: pl.Sequence
    state_dict: dict[str, int]
    target: int

    def __post_init__(self) -> None:
        self.state_dict = _convert_np_int64_to_int(data=self.state_dict)

    def save_to_file(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            tmp_dict = {
                "sequence": self.sequence.to_abstract_repr(),
                "state_dict": self.state_dict,
                "target": self.target,
            }
            json.dump(tmp_dict, file)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ProcessedData":
        with open(file_path) as file:
            tmp_data = json.load(file)
            return cls(
                sequence=pl.Sequence.from_abstract_repr(obj_str=tmp_data["sequence"]),
                state_dict=tmp_data["state_dict"],
                target=tmp_data["target"],
            )


def _convert_np_int64_to_int(data: dict[str, np.int64]) -> dict[str, int]:
    return {
        key: (int(value) if isinstance(value, np.integer) else value) for key, value in data.items()
    }
