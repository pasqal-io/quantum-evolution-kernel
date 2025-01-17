from __future__ import annotations

import collections
import json
import matplotlib

import numpy as np
import pulser as pl


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
      corresponding qubit.
    - values are the number of samples observed with this specific
      state of the register.

    The sum of all values for the dictionary is equal to the total
    number of samples observed on the quantum device (for this
    specific graph).
    """

    sequence: pl.Sequence
    state_dict: dict[str, int]
    _dist_excitation: np.ndarray
    target: int

    def __init__(self, sequence: pl.Sequence, state_dict: dict[str, np.int64], target: int):
        self.sequence = sequence
        self.state_dict = _convert_np_int64_to_int(data=state_dict)
        self._dist_excitation = dist_excitation(self.state_dict)
        self.target = target

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

    def dist_excitation(self, width: int | None = None) -> np.ndarray:
        """
        Return the distribution of excitations for this graph.

        Arguments:
            width: If specified, truncate or extend the result to this width.
        """
        if width is None or width == len(self._dist_excitation):
            return self._dist_excitation.copy()
        if width < len(self._dist_excitation):
            return np.resize(self._dist_excitation, width)
        return np.pad(self._dist_excitation, (0, width - len(self._dist_excitation)))

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

    def draw_excitation(self) -> None:
        """
        Draw an histogram for the excitation level on screen
        """
        x = [str(i) for i in range(len(self._dist_excitation))]
        matplotlib.pyplot.bar(x, self._dist_excitation)


def dist_excitation(state_dict: dict[str, int], size_max: int | None = None) -> np.ndarray:
    """
    Calculates the distribution of excitation energies from a dictionary of
    bitstrings to their respective counts.

    Args:
        size_max (int | None): If specified, only keep `size_max` energy
            distributions in the output. Otherwise, keep all values.

    Returns:
        A histogram of excitation energies.
        - index: an excitation level (i.e. a number of `1` bits in a
            bitstring)
        - value: normalized count of samples with this excitation level.
    """

    if len(state_dict) == 0:
        return np.ndarray()

    if size_max is None:
        # If size is not specified, it's the length of bitstrings.
        # We assume that all bitstrings in `count_bitstring` have the
        # same length and we have just checked that it's not empty.

        # Pick the length of the first bitstring.
        # We have already checked that `count_bitstring` is not empty.
        bitstring = next(iter(state_dict.keys()))
        size_max = len(bitstring)

    # Make mypy realize that `size_max` is now always an `int`.
    assert type(size_max) is int

    count_occupation: dict[int, int] = collections.defaultdict(int)
    total = 0.0
    for bitstring, number in state_dict.items():
        occupation = sum(1 for bit in bitstring if bit == "1")
        count_occupation[occupation] += number
        total += number

    result = np.zeros(size_max + 1, dtype=float)
    for occupation, count in count_occupation.items():
        if occupation < size_max:
            result[occupation] = count / total

    return result


def _convert_np_int64_to_int(data: dict[str, np.int64]) -> dict[str, int]:
    """
    Utility function: convert the values of a dict from `np.int64` to `int`,
    for serialization purposes.
    """
    return {
        key: (int(value) if isinstance(value, np.integer) else value) for key, value in data.items()
    }
