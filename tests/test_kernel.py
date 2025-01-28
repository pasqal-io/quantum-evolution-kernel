import qek.data.dataset as qek_dataset
from qek.kernel import QuantumEvolutionKernel


def test_kernel() -> None:
    """
    A few basic checks on QuantumEvolutionKernel.
    """

    # Load dataset
    processed_dataset = qek_dataset.load_dataset(file_path="examples/ptcfm_processed_dataset.json")

    # Test with various qubit lengths.
    #
    # We expect that a size_max of 5000 qubits should be sufficient to be larger than the number of qubits needed to
    # execute our dataset.
    for size_max in [None, 0, 5000]:
        qek = QuantumEvolutionKernel(mu=2.0, size_max=size_max)
        similarities = qek(processed_dataset, processed_dataset)
        assert len(similarities) == len(processed_dataset)
        assert len(similarities[0]) == len(processed_dataset)
        for i in range(len(similarities)):
            assert (
                similarities[i, i] >= 0.999
            )  # It should be 1, but let's allow for rounding errors.
            for j in range(len(similarities)):
                print(f"similarities[{i}, {j}] == {similarities[i, j]}")
                assert (
                    abs(similarities[i, j] - similarities[j, i]) < 0.001
                )  # It should be 0, but let's allow for rounding errors.
                assert similarities[i, j] >= 0
                assert (
                    similarities[i, j] <= 1.001
                )  # It should be 1, but let's allow for rounding errors.
