from __future__ import annotations

from typing import Any
import collections
import copy
from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon

from qek.data.dataset import ProcessedData


class QuantumEvolutionKernel:
    """QuantumEvolutionKernel class.

    Attributes:
    - params (dict): Dictionary of training parameters.
    - X (Sequence[ProcessedData]): Training data used for fitting the kernel
    - kernel_matrix (np.ndarray): Kernel matrix. This is assigned in the `fit()` method


    """

    def __init__(self, mu: float):
        """Initialize the QuantumEvolutionKernel.

        Args:
            mu (float): Scaling factor for the Jensen-Shannon divergence
        """
        self.params: dict[str, Any] = {"mu": mu}
        self.X: Sequence[ProcessedData]
        self.kernel_matrix: np.ndarray

    def __call__(
        self, graph_1: ProcessedData, graph_2: ProcessedData, size_max: int | None = None
    ) -> float:
        """Compute the similarity between two graphs using Jensen-Shannon
        divergence.

        This method computes the square of the Jensen-Shannon divergence (JSD)
        between two probability distributions over bitstrings. The JSD is a
        measure of the difference between two probability distributions, and it
        can be used as a kernel for machine learning algorithms that require a
        similarity function.

        The input graphs are assumed to have been processed using the
        ProcessedData class from qek_os.data_io.dataset. Parameter `size_max`
        controls the maximum length of the bitstrings considered in the
        computation.
        Args:
            graph_1 (ProcessedData): First graph.
            graph_2 (ProcessedData): Second graph.
            size_max (int, optional): Maximum length of bitstrings to
            consider. Defaults to all.

        Returns:
            float: Similarity between the two graphs, scaled by a factor that
            depends on mu.

        Notes:
            The JSD is computed using the jensenshannon function from
            `scipy.spatial.distance`, and it is squared because scipy function
            `jensenshannon` outputs the distance instead of the divergence.
        """
        if len(graph_1.state_dict) == 0 or len(graph_2.state_dict) == 0:
            raise ValueError("An input counter is empty")

        if size_max is None:
            # If size is not specified, it's the length of bitstrings.
            bitstring_1 = next(iter(graph_1.state_dict.keys()))
            bitstring_2 = next(iter(graph_1.state_dict.keys()))
            size_max = max(len(bitstring_1), len(bitstring_2))

        dist_graph_1 = dist_excitation_and_vec(
            count_bitstring=graph_1.state_dict, size_max=size_max
        )
        dist_graph_2 = dist_excitation_and_vec(
            count_bitstring=graph_2.state_dict, size_max=size_max
        )
        js = (
            jensenshannon(p=dist_graph_1, q=dist_graph_2) ** 2
        )  # Because the divergence is the square root of the distance
        return float(np.exp(-self.params["mu"] * js))

    def fit(self, X: Sequence[ProcessedData], y: list | None = None) -> None:
        """Fit the kernel to the training dataset by storing the dataset.

        Args:
            X (Sequence[ProcessedData]): The training dataset.
            y: list: Target variable for the dataset sequence. defaults to None.
        """
        self.X = X
        self.kernel_matrix = self.create_train_kernel_matrix(self.X)

    def transform(self, X_test: Sequence[ProcessedData], y_test: list | None = None) -> np.ndarray:
        """Transform the dataset into the kernel space with respect to the training dataset.

        Args:
            X_test (Sequence[ProcessedData]): The dataset to transform.
            y_test: list: Target variable for the dataset sequence. defaults to None.

        Returns:
            np.ndarray: Kernel matrix where each entry represents the similarity between
                        the given dataset and the training dataset.
        """
        if self.X is None:
            raise ValueError("The kernel must be fit to a training dataset before transforming.")

        return self.create_test_kernel_matrix(X_test, self.X)

    def fit_transform(self, X: Sequence[ProcessedData], y: list | None = None) -> np.ndarray:
        """Fit the kernel to the training dataset and transform it.

        Args:
            X (Sequence[ProcessedData]): The dataset to fit and transform.
            y: list: Target variable for the dataset sequence. defaults to None.

        Returns:
            np.ndarray: Kernel matrix for the training dataset.
        """
        self.fit(X)
        return self.kernel_matrix

    def create_train_kernel_matrix(self, train_dataset: Sequence[ProcessedData]) -> np.ndarray:
        """Compute a kernel matrix for a given training dataset.

        This method computes a symmetric N x N kernel matrix from the
        Jensen-Shannon divergences between all pairs of graphs in the input
        dataset. The resulting matrix can be used as a similarity metric for
        machine learning algorithms.
        Args:
            train_dataset (Sequence[ProcessedData]): A list of ProcessedData
            objects to compute the kernel matrix from.
        Returns:
            np.ndarray: An N x N symmetric matrix where the entry at row i and
            column j represents the similarity between the graphs in positions
            i and j of the input dataset.
        """
        N = len(train_dataset)
        kernel_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                kernel_mat[i][j] = self(train_dataset[i], train_dataset[j])
                kernel_mat[j][i] = kernel_mat[i][j]
        return kernel_mat

    def create_test_kernel_matrix(
        self,
        test_dataset: Sequence[ProcessedData],
        train_dataset: Sequence[ProcessedData],
    ) -> np.ndarray:
        """
        Compute a kernel matrix for a given testing dataset and training
        set.

        This method computes an N x M kernel matrix from the Jensen-Shannon
        divergences between all pairs of graphs in the input testing dataset
        and the training dataset.
        The resulting matrix can be used as a similarity metric for machine
        learning algorithms,
        particularly when evaluating the performance on the test dataset using
        a trained model.
        Args:
            test_dataset (Sequence[ProcessedData]): A list of ProcessedData
            objects representing the testing dataset.
            train_dataset (Sequence[ProcessedData]): A list of ProcessedData
            objects representing the training set.
        Returns:
            np.ndarray: An M x N matrix where the entry at row i and column j
            represents the similarity between the graph in position i of the
            test dataset and the graph in position j of the training set.
        """
        N_train = len(train_dataset)
        N_test = len(test_dataset)
        kernel_mat = np.zeros((N_test, N_train))
        for i in range(N_test):
            for j in range(N_train):
                kernel_mat[i][j] = self(test_dataset[i], train_dataset[j])
        return kernel_mat

    def set_params(self, **kwargs: dict[str, Any]) -> None:
        """Set multiple parameters for the kernel.

        Args:
            **kwargs: Arbitrary keyword dictionary where keys are attribute names
            and values are their respective values
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_params(self, deep: bool = True) -> dict:
        """Retrieve the value of all parameters.

         Args:
            deep (bool): Ignored. Added for compatibility with various machine learning libraries,
                such as scikit-learn.

        Returns
            dict: A dictionary of parameters and their respective values.
        """
        return copy.deepcopy(self.params)


def count_occupation_from_bitstring(bitstring: str) -> int:
    """Counts the number of '1' bits in a binary string.

    Args:
        bitstring (str): A binary string containing only '0's and '1's.

    Returns:
        int: The number of '1' bits found in the input string.
    """
    return sum(int(bit) for bit in bitstring)


def dist_excitation_and_vec(
    count_bitstring: dict[str, int], size_max: int | None = None
) -> np.ndarray:
    """
    Calculates the distribution of excitation energies from a dictionary of
    bitstrings to their respective counts.

    Args:
        count_bitstring (dict[str, int]): A dictionary mapping binary strings
            to their counts.
        size_max (int | None): If specified, only keep `size_max` energy
            distributions in the output. Otherwise, keep all values.

    Returns:
        np.ndarray: A NumPy array where keys are the number of '1' bits
            in each binary string and values are the normalized counts.
    """

    if len(count_bitstring) == 0:
        raise ValueError("The input counter is empty")

    if size_max is None:
        # If size is not specified, it's the length of bitstrings.
        # We assume that all bitstrings in `count_bitstring` have the
        # same length and we have just checked that it's not empty.

        # Pick the length of the first bitstring.
        # We have already checked that `count_bitstring` is not empty.
        bitstring = next(iter(count_bitstring.keys()))
        size_max = len(bitstring)

    # Make mypy realize that `size_max` is now always an `int`.
    assert type(size_max) is int

    count_occupation: dict[int, int] = collections.defaultdict(int)
    total = 0.0
    for k, v in count_bitstring.items():
        occupation = count_occupation_from_bitstring(k)
        count_occupation[occupation] += v
        total += v

    numpy_vec = np.zeros(size_max + 1, dtype=float)
    for occupation, count in count_occupation.items():
        if occupation < size_max:
            numpy_vec[occupation] = count / total

    return numpy_vec
