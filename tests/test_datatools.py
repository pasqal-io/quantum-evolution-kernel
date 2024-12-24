from __future__ import annotations

import torch_geometric.datasets as pyg_dataset

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

    # Some of the graphs from the original data can be converted in disk graphs
    # by the algorithm, but not all.
    #
    # As we know that the first graph can be, we'll use it to test against
    # regressions.
    augmented_graph = add_graph_coord(graph=original_ptcfm_data[0],
                                      blockade_radius=RADIUS)
    assert is_disk_graph(augmented_graph, RADIUS + EPS)
