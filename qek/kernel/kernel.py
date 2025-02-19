"""
The Quantum Evolution Kernel itself, for use in a machine-learning pipeline.
"""

from __future__ import annotations

from typing import Any, Callable
import collections
import copy
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import jensenshannon

from qek.data.processed_data import ProcessedData


class QuantumEvolutionKernel:
    """Implementation of the Quantum Evolution Kernel.

    Attributes:
    - params (dict): Dictionary of training parameters. As of this writing, the only
        training parameter is "mu", the scaling factor for the Jensen-Shannon divergence.
    - X (Sequence[ProcessedData]): Training data used for fitting the kernel.
    - kernel_matrix (np.ndarray): Kernel matrix. This is assigned in the `fit()` method


    """

    def __init__(
        self,
        mu: float,
        size_max: int | None = None,
        similarity: (
            Callable[[NDArray[np.floating], NDArray[np.floating]], np.floating] | None
        ) = None,
    ):
        """Initialize the QuantumEvolutionKernel.

        Args:
            mu (float): Scaling factor for the Jensen-Shannon divergence
            size_max (int, optional): If specified, only consider the first `size_max`
                qubits of bitstrings. Otherwise, consider all qubits. You may use this
                to trade precision in favor of speed.
            similarity (optional): If specified, a custom similarity metric to use. Otherwise,
                use the Jensen-Shannon divergence.
        """
        self.params: dict[str, Any] = {
            "mu": mu,
            "size_max": size_max,
            "similarity": similarity if similarity is not None else self.js_similarity,
        }
        self.X: Sequence[ProcessedData]
        self.kernel_matrix: np.ndarray

    def __call__(
        self,
        X1: Sequence[ProcessedData],
        X2: Sequence[ProcessedData] | None = None,
    ) -> NDArray[np.floating]:
        """Compute a kernel matrix from two sequences of processed data.

        This method computes a M x N kernel matrix from the Jensen-Shannon divergences
        between all pairs of graphs in the two datasets. The resulting matrix can be used
        as a similarity metric for machine learning algorithms.

        If `X1` and `X2` are two sequences representing the processed data for a
        single graph each, the resulting matrix can be used as a measure of similarity
        between both graphs.

        Args:
            X1: processed data to be used as rows.
            X2 (optional): processed data to be used as columns. If unspecified, use X1
                as both rows and columns.
        Returns:
            np.ndarray: A len(X1) x len(X2) matrix where entry[i, j] represents the
            similarity between rows[i] and columns[j], scaled by a factor that depends
            on mu.
        Notes:
            The JSD is computed using the jensenshannon function from
            `scipy.spatial.distance`, and it is squared because scipy function
            `jensenshannon` outputs the distance instead of the divergence.
        """
        # If size is not specified, set it to the length of the largest bitstring.
        size_max = self.params["size_max"]
        if size_max is None:
            if X2 is None:
                # No need to walk the same source twice.
                sources = [X1]
            else:
                sources = [X1, X2]
            for source in sources:
                for data in source:
                    length = len(data._sequence.qubit_info)
                    if size_max is None or size_max <= length:
                        size_max = length

        # Note: At this stage, size_max could theoretically still be `None``, if both `X1` and `X2`
        # are empty. In such cases, `dist_excitation` will never be called, so we're ok.
        feat_rows = [row.dist_excitation(size_max) for row in X1]
        similarity: Callable[[NDArray[np.floating], NDArray[np.floating]], np.floating] = (
            self.params["similarity"]
        )
        if X2 is None:
            # Fast path:
            # - rows and columns are identical, so no need to compute a `feat_cols`;
            # - the matrix is symmetric, we only need to compute half of it.
            #
            # We could avoid computing kernel[i, i], as we know that it's always 1,
            # but we do not perform this specific optimization, as it is a useful
            # canary to detect some bugs.
            kernel = np.zeros([len(X1), len(X1)])
            for i, dist_row in enumerate(feat_rows):
                for j in range(i, len(feat_rows)):
                    dist_col = feat_rows[j]
                    s = similarity(dist_row, dist_col)
                    kernel[i, j] = s
                    if j != i:
                        kernel[j, i] = s
        else:
            # Slow path:
            # - we need to compute a `feat_columns`
            # - the matrix is generally not symmetric and diagonal entries are generally not 1.
            kernel = np.zeros([len(X1), len(X2)])
            feat_columns = [col.dist_excitation(size_max) for col in X2]
            for i, dist_row in enumerate(feat_rows):
                for j, dist_col in enumerate(feat_columns):
                    kernel[i, j] = similarity(dist_row, dist_col)
        return kernel

    def js_similarity(self, row: NDArray[np.floating], col: NDArray[np.floating]) -> np.floating:
        """
        The Jensen-Shannon similarity metric used to compute the kernel, used when calling `kernel(X1, X2)`.
        """
        js = jensenshannon(row, col) ** 2
        mu = float(self.params["mu"])
        return np.exp(-mu * js)

    def similarity(self, graph_1: ProcessedData, graph_2: ProcessedData) -> float:
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
        matrix = self([graph_1], [graph_2])
        return float(matrix[0, 0])

    def fit(self, X: Sequence[ProcessedData], y: list | None = None) -> None:
        """Fit the kernel to the training dataset by storing the dataset.

        Args:
            X (Sequence[ProcessedData]): The training dataset.
            y: list: Target variable for the dataset sequence.
                This argument is ignored, provided only for compatibility
                with machine-learning libraries.
        """
        self.X = X
        self.kernel_matrix = self.create_train_kernel_matrix(self.X)

    def transform(self, X_test: Sequence[ProcessedData], y_test: list | None = None) -> np.ndarray:
        """Transform the dataset into the kernel space with respect to the training dataset.

        Args:
            X_test (Sequence[ProcessedData]): The dataset to transform.
            y_test: list: Target variable for the dataset sequence.
                This argument is ignored, provided only for compatibility
                with machine-learning libraries.
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
            y: list: Target variable for the dataset sequence.
                This argument is ignored, provided only for compatibility
                with machine-learning libraries.
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
        return self(train_dataset)

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
        return self(test_dataset, train_dataset)

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
            deep (bool): Ignored for the time being. Added for compatibility with
                various machine learning libraries, such as scikit-learn.

        Returns
            dict: A dictionary of parameters and their respective values.
                Note that this method always performs a copy of the dictionary.
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
