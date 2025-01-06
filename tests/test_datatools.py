from __future__ import annotations

import networkx as nx
import torch_geometric.datasets as pyg_dataset
import torch_geometric.utils as pyg_utils

from qek.data.datatools import add_graph_coord
from qek.utils import is_disk_graph


def test_add_graph_coord() -> None:
    # Load dataset
    original_ptcfm_data = pyg_dataset.TUDataset(root="dataset", name="PTC_FM")

    # Check that `add_graph_coord` doesn't break with this dataset.
    RADIUS = 5.001
    EPS = 0.01

    for graph in original_ptcfm_data:
        augmented_graph = add_graph_coord(graph=graph, blockade_radius=RADIUS)

        # Make sure that the graph has been augmented with "pos".
        assert hasattr(augmented_graph, "pos")

        # Confirm that the augmented graph is isomorphic to the original graph.
        nx_graph = pyg_utils.to_networkx(
            data=graph,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
            to_undirected=True,
        )
        nx_reconstruct = pyg_utils.to_networkx(augmented_graph).to_undirected()

        assert nx.is_isomorphic(nx_graph, nx_reconstruct)

    # The first graph from the dataset is known to be a disk graph.
    augmented_graph = add_graph_coord(graph=original_ptcfm_data[0], blockade_radius=RADIUS)
    assert is_disk_graph(augmented_graph, RADIUS + EPS)
