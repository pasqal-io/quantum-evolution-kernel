"""
Loading graphs as raw data.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import logging
from typing import Any, Final, Generic, TypeVar

import networkx as nx
import numpy as np
import pulser as pl
import rdkit.Chem as Chem
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from rdkit.Chem import AllChem

from qek.shared.error import CompilationError
from qek.shared._utils import graph_to_mol

logger = logging.getLogger(__name__)

EPSILON_RADIUS_UM = 0.01
"""
Assumption of rounding error when determining whether a graph is a disk graph.
"""

EPSILON_RESCALE_FACTOR = 1.000000001
"""
A correction factor, attempting to cover for rounding error when rescaling a graph.
"""


class BaseGraph:
    """
    A graph being prepared for embedding on a quantum device.
    """

    device: Final[pl.devices.Device]

    def __init__(
        self, id: int, data: pyg_data.Data, device: pl.devices.Device, target: int | None = None
    ):
        """
        Create a graph from geometric data.

        Args:
            id: An identifier for this graph, used mostly for error messages.
            data:  A homogeneous graph, in PyTorch Geometric format. Unchanged.
                It MUST have attributes 'pos'.
            device: The device for which the graph is prepared.
        """
        if not hasattr(data, "pos"):
            raise AttributeError("The graph should have an attribute 'pos'.")

        # The device for which the graph is prepared.
        self.device = device

        # The graph in torch geometric format.
        self.pyg = data.clone()

        # The graph in networkx format, undirected.
        self.nx_graph: nx.Graph = pyg_utils.to_networkx(
            data=data,
            node_attrs=["x"],
            edge_attrs=["edge_attr"] if data.edge_attr is not None else None,
            to_undirected=True,
        )
        self.target = target
        self.id = id

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
            logger.debug("graph %s doesn't have any nodes, it's not a disk graph", self.id, self.id)
            return False

        # Check if the graph is connected.
        if len(self.nx_graph) == 0 or not nx.is_connected(self.nx_graph):
            logger.debug("graph %s is not connected, it's not a disk graph", self.id)
            return False

        # Check the distances between all pairs of nodes.
        pos = self.pyg.pos
        assert pos is not None
        for u, v in nx.non_edges(self.nx_graph):
            distance_um = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance_um <= radius:
                # These disjointed nodes would interact with each other, so
                # this is not an embeddable graph.
                logger.debug(
                    "graph %s has non-edges that are too close to each other, it's not a disk graph",
                    self.id,
                )
                return False

        for u, v in self.nx_graph.edges():
            distance_um = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance_um > radius:
                # These joined nodes would not interact with each other, so
                # this is not an embeddable graph.
                logger.debug(
                    "graph %s has edges that are too distant from each other (%s > %s), it's not a disk graph",
                    self.id,
                    distance_um,
                    radius,
                )
                return False

        return True

    def is_embeddable(self) -> bool:
        """
            A predicate to check if the graph can be embedded in the
            quantum device.

            For a graph to be embeddable on a device, all the following
            criteria must be fulfilled:
            - the graph must be non-empty;
            - the device must have at least as many atoms as the graph has
                nodes;
            - the device must be physically large enough to place all the
                nodes (device.max_radial_distance);
            - the nodes must be distant enough that quantum interactions
                may take place (device.min_atom_distance)

        Returns:
            bool: True if possible, False if not
        """

        # Reject empty graphs.
        if self.pyg.num_nodes == 0 or self.pyg.num_nodes is None:
            logger.debug("graph %s is empty, it's not embeddable", self.id)
            return False

        # Reject graphs that have more nodes than can be represented
        # on the device.
        if self.pyg.num_nodes > self.device.max_atom_num:
            logger.debug(
                "graph %s has too many nodes (%s), it's not embeddable", self.id, self.pyg.num_nodes
            )
            return False

        # Check the distance from the center
        pos = self.pyg.pos
        assert pos is not None
        distance_from_center = np.linalg.norm(pos, ord=2, axis=-1)
        if any(distance_from_center > self.device.max_radial_distance):
            logger.debug(
                "graph %s has nodes to far from the center (%s > %s), it's not embeddable",
                self.id,
                max(distance_from_center),
                self.device.max_radial_distance,
            )
            return False

        # Check distance between nodes
        if not self.is_disk_graph(self.device.min_atom_distance + EPSILON_RADIUS_UM):
            logger.debug("graph %s is not a disk graph, it's not embeddable", self.id)
            return False

        for u, v in self.nx_graph.edges():
            distance_um = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance_um < self.device.min_atom_distance:
                # These nodes are too close to each other, preventing quantum
                # interactions on the device.
                logger.debug(
                    "graph %s has nodes that are too close to each other (%s < %s), it's not embeddable",
                    self.id,
                    distance_um,
                    self.device.min_atom_distance,
                )
                return False

        return True

    # Default values for the sequence.
    #
    # See the companion paper for an explanation.
    SEQUENCE_DEFAULT_AMPLITUDE_RAD_PER_US = 1.0 * 2 * np.pi
    SEQUENCE_DEFAULT_DURATION_NS = 660

    def compile_register(self) -> pl.Register:
        """Create a Quantum Register based on a graph.

        Returns:
            pulser.Register: register
        """
        # Note: In the low-level API, we separate register and pulse compilation for
        # pedagogical reasons, because we want to take the opportunity to teach them
        # about registers and pulses, rather than pulser sequences.

        if not self.is_embeddable():
            raise CompilationError(f"The graph is not compatible with {self.device}")

        # Compile register
        pos = self.pyg.pos
        assert pos is not None
        reg = pl.Register.from_coordinates(coords=pos)
        if self.device.requires_layout:
            reg = reg.with_automatic_layout(device=self.device)

        try:
            # Due to issue #29, we can produce a register that will not work on this device,
            # so we need to perform a second check.
            pl.Sequence(register=reg, device=self.device)
        except ValueError as e:
            raise CompilationError(f"The graph is not compatible with {self.device}: {e}")
        return reg

    def compile_pulse(
        self,
        amplitude: float = SEQUENCE_DEFAULT_AMPLITUDE_RAD_PER_US,
        duration: int = SEQUENCE_DEFAULT_DURATION_NS,
    ) -> pl.Pulse:
        """Extract a Pulse for this graph.

        A Pulse represents the laser applied to the atoms on the device.

        Arguments:
            amplitude: The amplitude for the laser pulse, in rad per microseconds.
                By default, use the value demonstrated in the companion paper.
            duration: The duration of the laser pulse, in nanoseconds.
                By default, use the value demonstrated in the companion paper.
        """
        # Note: In the low-level API, we separate register and pulse compilation for
        # pedagogical reasons, because we want to take the opportunity to teach them
        # about registers and pulses, rather than pulser sequences.

        # See the companion paper for an explanation on these constants.
        Omega_max = amplitude
        t_max = duration
        pulse = pl.Pulse.ConstantAmplitude(
            amplitude=Omega_max,
            detuning=pl.waveforms.RampWaveform(t_max, 0, 0),
            phase=0.0,
        )
        return pulse


class MoleculeGraph(BaseGraph):
    """
    A graph based on molecular data, being prepared for embedding on a
    quantum device.
    """

    def __init__(
        self,
        id: Any,
        data: pyg_data.Data,
        device: pl.devices.Device,
        node_mapping: dict[int, str],
        edge_mapping: dict[int, Chem.BondType],
        target: int | None = None,
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
            target: If specified, a target for machine learning, as a value
                `0` or `1`.
        """
        pyg = data.clone()
        pyg.pos = None  # Placeholder
        super().__init__(id=id, data=pyg, device=device, target=target)

        # Reconstruct the molecule.
        tmp_mol = graph_to_mol(
            graph=self.nx_graph,
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
        )

        # Extract the geometry.
        AllChem.Compute2DCoords(tmp_mol, useRingTemplates=True)
        original_pos = tmp_mol.GetConformer().GetPositions()[..., :2]  # Convert to 2D

        # We now want to scale the geometry so that the smallest edge
        # is as long as `device.min_atom_distance`.
        pos = original_pos
        pairs: list[tuple[Any, Any]] = []
        for start, end in self.nx_graph.edges():
            pairs.append((start, end))
        for start, end in nx.non_edges(self.nx_graph):
            pairs.append((start, end))

        distances = []
        for start, end in pairs:
            distances.append(np.linalg.norm(pos[start] - pos[end]))
        min_distance = np.min(distances)
        pos = pos * device.min_atom_distance / min_distance

        # The above transformation is sensitive to rouding errors, so if we realize that
        # we accidentally made the smallest edge too small, we'll multiply by a small factor.
        while True:
            distances = []
            for start, end in pairs:
                distances.append(np.linalg.norm(pos[start] - pos[end]))
            min_distance = np.min(distances)
            if min_distance >= device.min_atom_distance:
                logger.debug(
                    "The minimal distance in graph #%s exceeds min atom distance: %s > %s, it should now be compilable",
                    self.id,
                    min_distance,
                    device.min_atom_distance,
                )
                break
            pos = pos * EPSILON_RESCALE_FACTOR

        # Finally, store the position.
        self.pyg.pos = pos


class PTCFMGraph(MoleculeGraph):
    """
    An ingester for molecule graphs using
    PTC-FM dataset conventions.
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
        id: Any,
        data: pyg_data.Data,
        device: pl.devices.Device,
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
            target: If specified, a target for machine learning, as a value
                `0` or `1`.
        """
        target = data.y
        if target is None:
            raise AttributeError("The graph should have an attribute 'y'.")

        if isinstance(target, torch.Tensor):
            target = target.item()
        target = int(target)

        super().__init__(
            id=id,
            data=data,
            device=device,
            node_mapping=PTCFMGraph.PTCFM_ATOM_NAMES,
            edge_mapping=PTCFMGraph.PTCFM_BOND_TYPES,
            target=target,
        )


GraphType = TypeVar("GraphType")


class BaseGraphCompiler(abc.ABC, Generic[GraphType]):
    """
    Abstract class, used to load a graph and compile a Pulser sequence for a device.

    You should probably use one of the subclasses.
    """

    @abc.abstractmethod
    def ingest(self, graph: GraphType, device: pl.devices.Device, id: int) -> BaseGraph:
        raise Exception("Please use one of the subclasses")


class PygWithPosCompiler(BaseGraphCompiler[pyg_data.Data]):
    """
    A compiler able to ingest torch_geometric graphs with positions.
    """

    def ingest(self, graph: pyg_data.Data, device: pl.devices.Device, id: int) -> BaseGraph:
        """
        Compile a Pulser sequence from a torch_geometric graph with position.

        Args:
            graph: A graph with positions (specified as attribute `pos`) and
                optionally a prediction target (specified as attribute `y`, which
                must be an `int`). The graph will not be changed.
            device: The device for which the sequence must be compiled.
            id: A unique identifier for the graph, used mostly for logging
                and displaying error messages.
        """
        return BaseGraph(id=id, data=graph, device=device)


class MoleculeGraphCompiler(BaseGraphCompiler[tuple[pyg_data.Data, int | None]]):
    """
    A compiler able to ingest torch_geometric molecules with a target.
    """

    def __init__(
        self,
        node_mapping: dict[int, str],
        edge_mapping: dict[int, Chem.BondType],
    ):
        """
        Setup a molecule graph compiler.

        Args:
            node_mapping: A mapping from node labels (as integers) to atom names (e.g. "C", "Ar", ...).
            edge_mapping: A mapping from node labels (as integers) to chemical bond types (e.g. simple
                bound, double bound).
        """
        self.node_mapping = node_mapping
        self.edge_mapping = edge_mapping

    """
    Compile a Pulser sequence from a molecule, expressed as a torch_geometric
    graph.

    Args:
        graph: A molecular graph.
            This graph is expected to have:
             - `int` labels on nodes, which may be converted into atom names
                with `self.node_mapping`
             - `int` labels on edges, which may be converted into chemical
                bounds with `self.edge_mapping`
            The graph will not be changed.
        device: The device for which the sequence must be compiled.
        id: A unique identifier for the graph, used mostly for logging
            and displaying error messages.
    """

    def ingest(
        self, graph: tuple[pyg_data.Data, int | None], device: pl.devices.Device, id: int
    ) -> MoleculeGraph:
        return MoleculeGraph(
            id=id,
            data=graph[0],
            device=device,
            node_mapping=self.node_mapping,
            edge_mapping=self.edge_mapping,
            target=graph[1],
        )


class PTCFMCompiler(BaseGraphCompiler[pyg_data.Data]):
    def ingest(self, graph: pyg_data.Data, device: pl.devices.Device, id: int) -> PTCFMGraph:
        return PTCFMGraph(
            id=id,
            data=graph,
            device=device,
        )


@dataclass
class NXWithPos:
    """
    A networkx graph and its position
    """

    graph: nx.Graph

    # Mapping from node to positions.
    positions: dict[Any, np.ndarray]

    # A machine learning target
    target: int | None


class NXGraphCompiler(BaseGraphCompiler[NXWithPos]):
    def ingest(self, graph: NXWithPos, device: pl.devices.Device, id: int) -> BaseGraph:
        pyg = pyg_utils.from_networkx(graph.graph)
        pyg.y = graph.target
        positions = np.array([graph.positions[node] for node in graph.graph.nodes()])
        pyg.pos = torch.tensor(positions, dtype=torch.float)

        return BaseGraph(id=id, device=device, data=pyg)
