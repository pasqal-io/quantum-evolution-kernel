from numpy import ndarray
import qek.data.dataset as qek_dataset


def test_excitation_distribution() -> None:
    """
    A few basic checks on excitation distribution functions.
    """

    # Load dataset
    processed_dataset = qek_dataset.load_dataset(file_path="examples/ptcfm_processed_dataset.json")

    for dataset in processed_dataset:
        # Check that the values we have make sense.
        excitation = dataset.dist_excitation()
        print(excitation)
        assert len(excitation) == len(dataset._sequence.qubit_info) + 1
        assert (excitation >= 0).all()
        assert (excitation <= 1).all()
        assert sum(excitation) >= 0.99
        assert sum(excitation) <= 1.01

    # Comparing excitation levels for a well-known dataset
    excitation = processed_dataset[0].dist_excitation()
    expected = ndarray(len(excitation), float)
    expected[0] = 0.958
    expected[1] = 0.041
    expected[2] = 0.0
    print(excitation)
    assert (excitation == expected).all()
