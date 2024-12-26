from __future__ import annotations

import torch
from torch_geometric.data import Data as Graph

from qek.utils import is_disk_graph


def test_is_disk_graph() -> None:
    # Empty graph
    graph_empty = Graph()
    assert not is_disk_graph(graph_empty, radius=1.0)

    # Single node
    graph_single_node = Graph(
        x=torch.tensor([0], dtype=torch.float),
        edge_index=torch.tensor([]),
    )
    assert is_disk_graph(graph_single_node, radius=1.0)

    # A graph with three nodes, close to each other, but one node
    # is not connected.
    graph_disconnected_close = Graph(
        x=torch.tensor([[0], [1], [2]], dtype=torch.float),
        edge_index=torch.tensor(
            [
                [0, 1],  # edge 0 -> 1
                [1, 0],  # edge 1 -> 0
            ],
            dtype=torch.int,
        ),
        pos=torch.tensor([[0], [1], [2]], dtype=torch.float),
    )
    assert not is_disk_graph(graph_disconnected_close, radius=10.0)

    # A graph with three nodes, close to each other, all nodes
    # are connected.
    graph_connected_close = Graph(
        x=torch.tensor([[0], [1], [2]], dtype=torch.float),
        edge_index=torch.tensor(
            [
                [
                    0,
                    1,  # edge 0 -> 1
                    1,
                    2,  # edge 1 -> 2
                    0,
                    2,  # edge 0 -> 2
                ],
                [
                    1,
                    0,  # edge 1 -> 0
                    2,
                    1,  # edge 2 -> 1
                    2,
                    0,  # edge 2 -> 0
                ],
            ],
            dtype=torch.int,
        ),
        pos=torch.tensor([[0], [1], [2]], dtype=torch.float),
    )
    assert is_disk_graph(graph_connected_close, radius=10.0)

    # A graph with three nodes, one is too far, all nodes
    # are connected.
    graph_connected_far = Graph(
        x=torch.tensor([[0], [1], [2]], dtype=torch.float),
        edge_index=torch.tensor(
            [
                [
                    0,
                    1,  # edge 0 -> 1
                    1,
                    2,  # edge 1 -> 2
                    0,
                    2,  # edge 0 -> 2
                ],
                [
                    1,
                    0,  # edge 1 -> 0
                    2,
                    1,  # edge 2 -> 1
                    2,
                    0,  # edge 2 -> 0
                ],
            ],
            dtype=torch.int,
        ),
        pos=torch.tensor([[0], [1], [12]], dtype=torch.float),
    )
    assert not is_disk_graph(graph_connected_far, radius=10.0)

    # A graph with three nodes, close to each other, one pair
    # of nodes is not connected.
    graph_partially_connected_close = Graph(
        x=torch.tensor([[0], [1], [2]], dtype=torch.float),
        edge_index=torch.tensor(
            [
                [
                    0,
                    1,  # edge 0 -> 1
                    1,
                    2,  # edge 1 -> 2
                ],
                [
                    1,
                    0,  # edge 1 -> 0
                    2,
                    1,  # edge 2 -> 1
                ],
            ],
            dtype=torch.int,
        ),
        pos=torch.tensor([[0], [1], [2]], dtype=torch.float),
    )
    assert not is_disk_graph(graph_partially_connected_close, radius=10.0)
