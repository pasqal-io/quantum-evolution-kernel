from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pulser as pl
import rdkit.Chem as Chem
import torch
import torch.utils.data as torch_data
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from rdkit.Chem import AllChem

from qek.data.conversion_data import PTCFM_EDGES_MAP, PTCFM_NODES_MAP
from qek.data.dataset import ProcessedData
from qek.utils import graph_to_mol


def add_graph_coord(
    graph: pyg_data.Data,
    blockade_radius: float,
    node_mapping: dict[int, str] = PTCFM_NODES_MAP,
    edge_mapping: dict[int, Chem.BondType] = PTCFM_EDGES_MAP,
) -> pyg_data.Data:
    """
    Take a molecule described as a graph with only nodes and edges,
    add 2D coordinates.

    This function:
    1. Converts the graph into a molecule (using `node_mapping` and
        `edge_mapping` to determine the types of atoms and bonds).
    2. Uses the molecule to determine coordinates.
    3. Injects the coordinates into the graph.

    :param pyg_data.Data graph:  A homogeneous graph, in PyTorch Geometric
        format. Unchanged.
    :param float blockade_radius: The radius of the Rydberg Blockade. Two
        connected nodes should be at a distance < blockade_radius, while
        two disconnected nodes should be at a distance > blockade_radius.
    :param node_mapping: A mapping of node labels from numbers to strings,
        e.g. `5 => "Cl"`. Used when building molecules.
    :param edge_mapping: A mapping of edge labels from number to chemical
        bond types, e.g. `2 => DOUBLE`. Used when building molecules.

    :return A clone of `graph` augmented with 2D coordinates.
    """
    graph = graph.clone()
    nx_graph = pyg_utils.to_networkx(
        data=graph,
        node_attrs=["x"],
        edge_attrs=["edge_attr"],
        to_undirected=True,
    )
    tmp_mol = graph_to_mol(
        graph=nx_graph,
        node_mapping=node_mapping,
        edge_mapping=edge_mapping,
    )
    AllChem.Compute2DCoords(tmp_mol, useRingTemplates=True)
    pos = tmp_mol.GetConformer().GetPositions()[..., :2]  # Convert to 2D
    dist_list = []
    for start, end in nx_graph.edges():
        dist_list.append(np.linalg.norm(pos[start] - pos[end]))
    norm_factor = np.max(dist_list)
    graph.pos = pos * blockade_radius / norm_factor
    return graph


def split_train_test(
    dataset: torch_data.Dataset,
    lengths: list[float],
    seed: int | None = None,
) -> tuple[torch_data.Dataset, torch_data.Dataset]:
    """
        This function splits a torch dataset into train and val dataset.
        As torch Dataset class is a mother class of pytorch_geometric dataset class,
        it should work just fine for the latter.
    Args:
        dataset (torch_data.Dataset): The original dataset to be splitted
        lengths (list[float]): Percentage of the split. For instance [0.8, 0.2]
        seed (int | None, optional): Seed for reproductibility. Defaults to None.

    Returns:
        tuple[torch_data.Dataset, torch_data.Dataset]: train and val dataset
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    train, val = torch_data.random_split(dataset=dataset, lengths=lengths, generator=generator)
    return train, val


def check_compatibility_graph_device(graph: pyg_data.Data, device: pl.devices.Device) -> bool:
    """Given the characteristics of a graph, return True if the graph can be embedded in
    the device, False if not.

    Args:
        graph (pyg_data): The graph to embeded
        device (pulser.devices.Device): the device

    Returns:
        bool: True if possible, False if not
    """
    pos_graph = graph.pos
    # check the number of atoms
    if graph.num_nodes > device.max_atom_num:
        return False
    # Check the distance from the center
    distance_from_center = np.linalg.norm(pos_graph, ord=2, axis=-1)
    if any(distance_from_center > device.max_radial_distance):
        return False
    if _return_min_dist(graph) < device.min_atom_distance:
        return False
    return True


def _return_min_dist(graph: pyg_data.Data) -> float:
    """Calculates the minimum distance between any two nodes in the graph, including
    both original and complementary edges.

    Args:
        graph (pyg_data.Data): The graph to calculate min distance from.

    Returns:
        float: Minimum distance between any two nodes.
    """
    nx_graph = pyg_utils.to_networkx(graph)
    graph_pos = graph.pos
    distances = []

    # get min distance in the graph
    for start, end in nx_graph.edges():
        distances.append(np.linalg.norm(graph_pos[start] - graph_pos[end], ord=2))
    compl_graph = nx.complement(nx_graph)
    for start, end in compl_graph.edges():
        distances.append(np.linalg.norm(graph_pos[start] - graph_pos[end], ord=2))
    min_dist: float = min(distances)
    return min_dist


def save_dataset(dataset: list[ProcessedData], file_path: str) -> None:
    """Saves a dataset to a JSON file.

    Args:
        dataset (list[ProcessedData]): The dataset to be saved, containing RegisterData instances.
        file_path (str): The path where the dataset will be saved as a JSON file.

    Note:
        The data is stored in a format suitable for loading with load_dataset.

    Returns:
        None
    """
    with open(file_path, "w") as file:
        data = [
            {
                "sequence": instance.sequence.to_abstract_repr(),
                "state_dict": instance.state_dict,
                "target": instance.target,
            }
            for instance in dataset
        ]
        json.dump(data, file)


def load_dataset(file_path: str) -> list[ProcessedData]:
    """Loads a dataset from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the dataset.

    Note:
        The data is loaded in the format that was used when saving with save_dataset.

    Returns:
        A list of ProcessedData instances, corresponding to the data stored in the JSON file.
    """
    with open(file_path) as file:
        data = json.load(file)
        return [
            ProcessedData(
                sequence=pl.Sequence.from_abstract_repr(item["sequence"]),
                state_dict=item["state_dict"],
                target=item["target"],
            )
            for item in data
        ]
