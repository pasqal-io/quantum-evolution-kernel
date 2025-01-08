from __future__ import annotations

import json
from typing import Final

import networkx as nx
import numpy as np
import pulser as pl
import rdkit.Chem as Chem
import torch
import torch.utils.data as torch_data
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from rdkit.Chem import AllChem

from qek.data.dataset import ProcessedData
from qek.utils import graph_to_mol


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


class BaseGraph:
    """
    A graph being prepared for embedding on a quantum device.
    """

    def __init__(self, data: pyg_data.Data):
        """
        Create a graph from geometric data.

        Args:
            data:  A homogeneous graph, in PyTorch Geometric format. Unchanged.
                It MUST have attributes 'pos'
        """
        if not hasattr(data, "pos"):
            raise AttributeError("The graph should have an attribute 'pos'.")

        # The graph in torch geometric format.
        self.pyg = data.clone()

        # The graph in networkx format, undirected.
        self.nx_graph = pyg_utils.to_networkx(
            data=data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if data.edge_attr is not None else None,
            to_undirected=True,
        )

    def is_disk_graph(self, radius: float) -> bool:
        """
        A predicate to check if `self` is a disk graph with the specified
        radius, i.e. `self` is a connected graph and, for every pair of nodes
        `A` and `B` within `graph`, there exists an edge between `A` and `B`
        if and only if the positions of `A` and `B` within `self` are such
        that `|AB| <= radius`.

        Args:
            radius: The maximal distance between two nodes of `self`
                connected be an edge.

        Returns:
            `True` if the graph is a disk graph with the specified radius,
            `False` otherwise.
        """

        if self.pyg.num_nodes == 0 or self.pyg.num_nodes is None:
            return False

        # Check if the graph is connected.
        if len(self.nx_graph) == 0 or not nx.is_connected(self.nx_graph):
            return False

        # Check the distances between all pairs of nodes.
        pos = self.pyg.pos
        for u, v in nx.non_edges(self.nx_graph):
            distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance <= radius:
                return False

        for u, v in self.nx_graph.edges():
            distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance > radius:
                return False

        return True

    def is_embeddable(self, device: pl.devices.Device) -> bool:
        """
            A predicate to check if the graph can be embedded in the
            quantum device.

            For a graph to be embeddable on a device, all the following
            criteria must be fulfilled:
            - the device must have at least as many atoms as the graph has
                nodes;
            - the device must be physically large enough to place all the
                nodes (device.max_radial_distance);
            - the nodes must be distant enough that quantum interactions
                may take place (device.min_atom_distance)

        Args:
            device (pulser.devices.Device): the device

        Returns:
            bool: True if possible, False if not
        """

        # Check the number of atoms
        if self.pyg.num_nodes > device.max_atom_num:
            return False

        # Check the distance from the center
        pos_graph = self.pyg.pos
        distance_from_center = np.linalg.norm(pos_graph, ord=2, axis=-1)
        if any(distance_from_center > device.max_radial_distance):
            return False

        # Check the distance between nodes.
        nodes = list(self.nx_graph.nodes)
        for i in range(0, len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = np.linalg.norm(pos_graph[i] - pos_graph[j], ord=2)
                if dist < device.min_atom_distance:
                    return False

        return True

    def compute_register(self) -> pl.Register:
        """Create a Quantum Register based on a graph.

        Returns:
            pulser.Register: register
        """
        return pl.Register.from_coordinates(coords=self.pyg.pos)

    def compute_sequence(self, device: pl.devices.Device) -> pl.Sequence:
        """
        Compile a Quantum Sequence from a graph for a specific device.

        Raises:
            ValueError if the graph cannot be embedded on the given device.
        """
        if not self.is_embeddable(device):
            raise ValueError(f"The graph is not compatible with {device}")
        reg = self.compute_register()
        seq = pl.Sequence(register=reg, device=device)

        # See the companion paper for an explanation on these constants.
        Omega_max = 1.0 * 2 * np.pi
        t_max = 660
        pulse = pl.Pulse.ConstantAmplitude(
            amplitude=Omega_max,
            detuning=pl.waveforms.RampWaveform(t_max, 0, 0),
            phase=0.0,
        )
        seq.declare_channel("ising", "rydberg_global")
        seq.add(pulse, "ising")
        return seq


class MoleculeGraph(BaseGraph):
    """
    A graph based on molecular data, being prepared for embedding on a
    quantum device.
    """

    # Constants used to decode the PTC-FM dataset, mapping
    # integers (used as node attributes) to atom names.
    PTCFM_ATOM_NAMES: Final[dict[int, str]] = {
        0: "In",
        1: "P",
        2: "C",
        3: "O",
        4: "N",
        5: "Cl",
        6: "S",
        7: "Br",
        8: "Na",
        9: "F",
        10: "As",
        11: "K",
        12: "Cu",
        13: "I",
        14: "Ba",
        15: "Sn",
        16: "Pb",
        17: "Ca",
    }

    # Constants used to decode the PTC-FM dataset, mapping
    # integers (used as edge attributes) to bond types.
    PTCFM_BOND_TYPES: Final[dict[int, Chem.BondType]] = {
        0: Chem.BondType.TRIPLE,
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.AROMATIC,
    }

    def __init__(
        self,
        data: pyg_data.Data,
        blockade_radius: float,
        node_mapping: dict[int, str] = PTCFM_ATOM_NAMES,
        edge_mapping: dict[int, Chem.BondType] = PTCFM_BOND_TYPES,
    ):
        """
        Compute the geometry for a molecule graph.

        Args:
            data:  A homogeneous graph, in PyTorch Geometric format. Unchanged.
            blockade_radius: The radius of the Rydberg Blockade. Two
                connected nodes should be at a distance < blockade_radius,
                while two disconnected nodes should be at a
                distance > blockade_radius.
            node_mapping: A mapping of node labels from numbers to strings,
                e.g. `5 => "Cl"`. Used when building molecules, e.g. to compute
                distances between nodes.
            edge_mapping: A mapping of edge labels from number to chemical
                bond types, e.g. `2 => DOUBLE`. Used when building molecules,
                e.g. to compute distances between nodes.
        """
        pyg = data.clone()
        pyg.pos = None  # Placeholder
        super().__init__(pyg)

        # Reconstruct the molecule.
        tmp_mol = graph_to_mol(
            graph=self.nx_graph,
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
        )

        # Extract the geometry.
        AllChem.Compute2DCoords(tmp_mol, useRingTemplates=True)
        pos = tmp_mol.GetConformer().GetPositions()[..., :2]  # Convert to 2D
        dist_list = []
        for start, end in self.nx_graph.edges():
            dist_list.append(np.linalg.norm(pos[start] - pos[end]))
        norm_factor = np.max(dist_list)

        # Finally, store the geometry.
        self.pyg.pos = pos * blockade_radius / norm_factor
