"""
Manipulating training data
"""

import torch
import torch.utils.data as torch_data


def split_train_test(
    dataset: torch_data.Dataset,
    lengths: list[float],
    seed: int | None = None,
) -> tuple[torch_data.Dataset, torch_data.Dataset]:
    """
        This function splits a torch dataset into train and val dataset.
        As torch Dataset class is a mother class of pytorch_geometric dataset
        class, it should work just fine for the latter.

    Args:
        dataset (torch_data.Dataset): The original dataset to be splitted
        lengths (list[float]): Percentage of the split. For instance [0.8, 0.2]
        seed (int | None, optional): Seed for reproductibility. Defaults to
        None.

    Returns:
        tuple[torch_data.Dataset, torch_data.Dataset]: train and val dataset
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    train, val = torch_data.random_split(dataset=dataset, lengths=lengths, generator=generator)
    return train, val
