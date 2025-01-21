from __future__ import annotations

from typing import Any
import collections
import copy
from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon

from qek.data.dataset import ProcessedData


class QuantumEvolutionKernel:
    def __init__(self, mu: float):
        self.params: dict[str, Any] = {"mu": mu}
        self.X: Sequence[ProcessedData]
        self.kernel_matrix: np.ndarray

    def __call__(
        self, graph_1: ProcessedData, graph_2: ProcessedData, size_max: int = 100
    ) -> float:
        """Compute the similarity between two graphs using Jensen-Shannon divergence.

        This method computes the square of the Jensen-Shannon divergence (JSD)
        between two probability distributions over bitstrings. The JSD is a
        measure of the difference between two probability distributions, and it
        can be used as a kernel for machine learning algorithms that require a
        similarity function.

        The input graphs are assumed to have been processed using the ProcessedData
        class from qek_os.data_io.dataset. The size_max parameter controls the maximum
        length of the bitstrings considered in the computation.
        Args:
            graph_1 (ProcessedData): First graph.
            graph_2 (ProcessedData): Second graph.
            size_max (float, optional): Maximum length of bitstrings to consider. Defaults to -1.

        Returns:
            float: Similarity between the two graphs, scaled by a factor that depends on mu.

        Notes:
            The JSD is computed using the jensenshannon function from scipy.spatial.distance,
            and it is squared because jensenshannon scipy function output the distance instead
            of the divergence.
        """
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

    def fit(self, X: Sequence[ProcessedData], y: list = None) -> None:
        """Fit the kernel to the training dataset by storing the dataset.

        Args:
            X (Sequence[ProcessedData]): The training dataset.
            y: list: Target variable for the dataset sequence. defaults to None.
        """
        self.X = X
        self.kernel_matrix = self.create_train_kernel_matrix(self.X)

    def transform(self, X_test: Sequence[ProcessedData], y_test: list = None) -> np.ndarray:
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

    def fit_transform(self, X: Sequence[ProcessedData], y: list = None) -> np.ndarray:
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

        This method computes a symmetric N x N kernel matrix from the Jensen-Shannon
        divergences between all pairs of graphs in the input dataset. The resulting matrix
        can be used as a similarity metric for machine learning algorithms.
        Args:
            train_dataset (Sequence[ProcessedData]): A list of ProcessedData objects to compute
            the kernel matrix from.
        Returns:
            np.ndarray: An N x N symmetric matrix where the entry at row i and column j represents
            the similarity between the graphs in positions i and j of the input dataset.
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
        """Compute a kernel matrix for a given testing dataset and training set.

        This method computes an N x M kernel matrix from the Jensen-Shannon
        divergences between all pairs of graphs in the input testing dataset
        and the training dataset.
        The resulting matrix can be used as a similarity metric for machine learning algorithms,
        particularly when evaluating the performance on the test dataset using a trained model.
        Args:
            test_dataset (Sequence[ProcessedData]): A list of ProcessedData
            objects representing the testing dataset.
            train_dataset (Sequence[ProcessedData]): A list of ProcessedData
            objects representing the training set.
        Returns:
            np.ndarray: An M x N matrix where the entry at row i and column j represents
            the similarity between the graph in position i of the test dataset
            and the graph in position j of the training set.
        """
        N_train = len(train_dataset)
        N_test = len(test_dataset)
        kernel_mat = np.zeros((N_test, N_train))
        for i in range(N_test):
            for j in range(N_train):
                kernel_mat[i][j] = self(test_dataset[i], train_dataset[j])
        return kernel_mat

    def set_params(self, **kwargs: dict[str, Any]) -> None:
        """Set multiple parameters dynamically using setattr

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


def dist_excitation_and_vec(count_bitstring: dict[str, int], size_max: int) -> np.ndarray:
    """Calculates the distribution of excitation energies from a dictionary of
    bitstrings to their respective counts, and then creates a NumPy vector with the
    results.

    Args:
        count_bitstring (dict[str, int]): A dictionary mapping binary strings
            to their counts.
        size_max (int): The maximum size of the resulting NumPy array.

    Returns:
        np.ndarray: A NumPy array where keys are the number of '1' bits
            in each binary string and values are the normalized counts.
    """
    count_occ: dict = collections.defaultdict(float)
    total = 0.0
    for k, v in count_bitstring.items():
        nbr_occ = count_occupation_from_bitstring(k)
        count_occ[nbr_occ] += v
        total += v

    numpy_vec = np.zeros(size_max)
    for k, v in count_occ.items():
        if int(k) <= size_max:
            numpy_vec[k] = v / total

    return numpy_vec
