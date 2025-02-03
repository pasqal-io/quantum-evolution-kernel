from typing import cast

import pulser
import networkx as nx
import torch
import torch_geometric.data as pyg_data
import torch_geometric.datasets as pyg_dataset
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

from qek.data.graphs import (
    BaseGraph,
    PTCFMCompiler,
    PTCFMGraph,
    PygWithPosCompiler,
    NXGraphCompiler,
    NXWithPos,
)


def test_graph_init() -> None:
    # Load dataset
    original_ptcfm_data = [
        cast(pyg_data.Data, d) for d in pyg_dataset.TUDataset(root="dataset", name="PTC_FM")
    ]

    # Check that `add_graph_coord` doesn't break with this dataset.

    for i, data in enumerate(original_ptcfm_data):
        graph = PTCFMGraph(data=data, device=pulser.AnalogDevice, id=i)

        # Make sure that the graph has been augmented with "pos".
        assert hasattr(graph.pyg, "pos")

        # Confirm that the augmented graph is isomorphic to the original graph.
        nx_graph = pyg_utils.to_networkx(
            data=data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
            to_undirected=True,
        )
        nx_reconstruct = pyg_utils.to_networkx(graph.pyg).to_undirected()

        assert nx.is_isomorphic(nx_graph, nx_reconstruct)

    # The first graph from the dataset is known to be embeddable.
    graph = PTCFMGraph(data=original_ptcfm_data[0], device=pulser.AnalogDevice, id=9999)
    assert graph.is_disk_graph(pulser.AnalogDevice.min_atom_distance + 0.01)


def test_is_disk_graph_false() -> None:
    """
    Testing is_disk_graph: these graphs are *not* disk graphs
    """
    # The empty graph is not a disk graph.
    graph_empty = BaseGraph(
        id=0,
        data=Data(
            x=torch.tensor([], dtype=torch.float),
            edge_index=torch.tensor([], dtype=torch.int),
            pos=torch.tensor([], dtype=torch.float),
            y=1,
        ),
        device=pulser.AnalogDevice,
    )
    assert not graph_empty.is_disk_graph(radius=1.0)

    # This graph has three nodes, each pair of nodes is closer than
    # the diameter, but it's not a disk graph because one of the nodes
    # is not connected.
    graph_disconnected_close = BaseGraph(
        id=0,
        data=Data(
            x=torch.tensor([[0], [1], [2]], dtype=torch.float),
            edge_index=torch.tensor(
                [
                    [0, 1],  # edge 0 -> 1
                    [1, 0],  # edge 1 -> 0
                ],
                dtype=torch.int,
            ),
            pos=torch.tensor([[0], [1], [2]], dtype=torch.float),
            y=1,
        ),
        device=pulser.AnalogDevice,
    )
    assert not graph_disconnected_close.is_disk_graph(radius=10.0)

    # This graph has three nodes, all nodes are connected, but it's
    # not a disk graph because one of the edges is longer than the
    # diameter.
    graph_connected_far = BaseGraph(
        id=0,
        data=Data(
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
            y=1,
        ),
        device=pulser.AnalogDevice,
    )
    assert not graph_connected_far.is_disk_graph(radius=10.0)

    # This graph has three nodes, each pair of nodes is within
    # the disk's diameter, but it's not a disk graph because
    # one of the pairs does not have an edge.
    graph_partially_connected_close = BaseGraph(
        id=0,
        data=Data(
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
            y=1,
        ),
        device=pulser.AnalogDevice,
    )
    assert not graph_partially_connected_close.is_disk_graph(radius=10.0)


def test_is_disk_graph_true() -> None:
    """
    Testing is_disk_graph: these graphs are disk graphs
    """
    # Single node
    graph_single_node = BaseGraph(
        id=0,
        data=Data(
            x=torch.tensor([0], dtype=torch.float),
            edge_index=torch.tensor([]),
            pos=torch.tensor([], dtype=torch.float),
            y=1,
        ),
        device=pulser.AnalogDevice,
    )
    assert graph_single_node.is_disk_graph(radius=1.0)

    # A complete graph with three nodes, each of the edges
    # is shorter than the disk's diameter.
    graph_connected_close = BaseGraph(
        id=0,
        data=Data(
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
            y=1,
        ),
        device=pulser.AnalogDevice,
    )
    assert graph_connected_close.is_disk_graph(radius=10.0)


def test_basic_compile() -> None:
    """
    Test basic properties of the various compilers. Mostly that they don't explode.
    """
    # Load dataset
    original_ptcfm_data = [
        cast(pyg_data.Data, d) for d in pyg_dataset.TUDataset(root="dataset", name="PTC_FM")
    ]

    # Testing PygWithPosCompiler
    pyg_with_pos_compiler = PygWithPosCompiler()
    pyg_with_pos_sample = Data(
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
        y=1,
    )

    ingested = pyg_with_pos_compiler.ingest(
        graph=pyg_with_pos_sample, device=pulser.AnalogDevice, id=999
    )
    saved_pyg_pos_ingested = ingested  # We'll reuse this graph
    assert ingested.id == 999
    assert len(ingested.nx_graph.nodes) == len(cast(torch.Tensor, pyg_with_pos_sample.x))
    assert len(ingested.nx_graph.edges) == len(cast(torch.Tensor, pyg_with_pos_sample.edge_index))

    # Testing NXGraphCompiler
    nx_graph_compiler = NXGraphCompiler()
    nx_graph = saved_pyg_pos_ingested.nx_graph
    pyg_graph_pos = saved_pyg_pos_ingested.pyg.pos
    assert pyg_graph_pos is not None  # Keep mypy happy
    list_positions = [pos for pos in pyg_graph_pos]
    list_nodes = [node for node in nx_graph.nodes()]
    dict_positions = {node: pos.numpy() for (node, pos) in zip(list_nodes, list_positions)}
    ingested = nx_graph_compiler.ingest(
        graph=NXWithPos(graph=nx_graph, positions=dict_positions, target=1),
        device=pulser.AnalogDevice,
        id=99,
    )
    assert ingested.id == 99
    assert nx.is_isomorphic(ingested.nx_graph, saved_pyg_pos_ingested.nx_graph)
    assert (ingested.pyg.pos == saved_pyg_pos_ingested.pyg.pos).all()

    # Testing MoleculeGraphCompiler
    molecule_graph_compiler = PTCFMCompiler()
    ingested = molecule_graph_compiler.ingest(
        graph=original_ptcfm_data[0], device=pulser.AnalogDevice, id=9
    )
    assert ingested.id == 9
