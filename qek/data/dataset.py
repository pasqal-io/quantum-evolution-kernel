from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pulser as pl


@dataclass
class ProcessedData:
    """
    Data on a single graph obtained from the Quantum Device.

    Attributes:
        sequence: The sequence, derived from the graph geometry,
            executed on the device.
        state_dict: A dictionary {bitstring: number of instances}
            for this graph.
        target: The target, i.e. the identifier of the graph.

    The state dictionary represents an approximation of the quantum
    state of the device for this graph after completion of the
    algorithm.

    - keys are bitstrings, i.e. strings of N time 0 or 1, where N
      is the number of qubits, i.e. the number of nodes in the graph.
      Each of these {0, 1} corresponds to a possible state for the
      corresponding qubit/node.
    - values are the number of samples observed with this specific
      state of the register/graph.

    The sum of all values for the dictionary is equal to the total
    number of samples observed on the quantum device for this graph.

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

    def draw_sequence(self) -> None:
        """
        Draw the sequence on screen
        """
        self.sequence.draw()

    def draw_register(self) -> None:
        """
        Draw the register on screen
        """
        self.sequence.register.draw(blockade_radius=self.sequence.device.min_atom_distance + 0.01)


def _convert_np_int64_to_int(data: dict[str, np.int64]) -> dict[str, int]:
    return {
        key: (int(value) if isinstance(value, np.integer) else value) for key, value in data.items()
    }
