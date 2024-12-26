from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
import pulser as pl
import rdkit.Chem as Chem
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils


def graph_to_mol(
    graph: nx.Graph,
    node_mapping: dict[int, str],
    edge_mapping: dict[int, Chem.BondType],
) -> Chem.Mol:
    """Reconstruct an rdkit mol object using a graph.

    Args:
        graph (nx.Graph): Networkx graph of a molecule.
        mapping (MolMapping): Object containing dicts for edges and nodes attributes.

    Returns:
        Chem.Mol: The generated rdkit molecule.
    """
    m = Chem.MolFromSmiles("")
    mw = Chem.RWMol(m)
    atom_index = {}
    for n, d in graph.nodes(data="x"):
        d = np.asarray(d)
        idx_d: int = inverse_one_hot(d, dim=0)[0]
        atom_index[n] = mw.AddAtom(Chem.Atom(node_mapping[idx_d]))
    for a, b, d in graph.edges(data="edge_attr"):
        start = atom_index[a]
        end = atom_index[b]
        d = np.asarray(d)
        idx_d = inverse_one_hot(d, dim=0)[0]
        bond_type = edge_mapping.get(idx_d)
        if bond_type is None:
            raise ValueError("bond type not implemented")
        mw.AddBond(start, end, bond_type)
    return mw.GetMol()


def inverse_one_hot(array: npt.ArrayLike, dim: int) -> np.ndarray:
    """
    Inverts a one-hot encoded tensor along a specified dimension and
    returns the indices where the value is 1.

    Parameters:
    - array (np.ndarray): The one-hot encoded array.
    - dim (int): The dimension along which to find the indices.

    Returns:
    - np.ndarray: The array of indices where the value is 1.
    """
    tmp_array = np.asarray(array)
    return np.nonzero(tmp_array == 1.0)[dim]


def is_disk_graph(graph: pyg_data.Data, radius: float) -> bool:
    """
    Check if `graph` is a disk graph with the specified radius, i.e.
    `graph` is a connected graph and, for every pair of nodes `A` and `B`
    within `graph`, there exists there exists an edge between `A` and `B`
    if and only if the positions of `A` and `B` within `graph` are such
    that `|AB| <= radius`.

    :param pyg_data.Data graph:  A homogeneous, undirected, graph, in PyTorch
        Geometric format. This graph MUST have an attribute `pos`, as provided
        e.g. by `datatools.add_graph_coord`.
    :param float radius: The maximal distance between two nodes of `graph`
        connected be an edge.

    :return bool `True` if the graph is a disk graph with the specified radius,
        `False` otherwise.
    """

    if hasattr(graph, "pos"):
        pos = graph.pos
    else:
        raise AttributeError("Graph object does not have a pos attribute")

    if graph.num_nodes == 0 or graph.num_nodes is None:
        return False

    # Molecule are unidrected Graphs.
    nx_graph = pyg_utils.to_networkx(graph, to_undirected=True)

    # Check if the graph is connected.
    if len(nx_graph) == 0 or not nx.is_connected(nx_graph):
        return False

    # Check the distances between all pairs of nodes.
    for u, v in nx.non_edges(nx_graph):
        distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        if distance <= radius:
            return False

    for u, v in nx_graph.edges():
        distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        if distance > radius:
            return False

    return True


def compute_register(data_graph: pyg_data.Data) -> pl.Register:
    """Create a register based on a graph using pulser.

    Args:
        data_graph (pyg_data.Data): graph. It should have a node attribute named UD_pos

    Returns:
        pulser.Register: register
    """
    if not hasattr(data_graph, "pos"):
        raise AttributeError("Graph should have a pos attribute")
    position = data_graph.pos
    return pl.Register.from_coordinates(coords=position)
