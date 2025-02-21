"""
Misc utility code that doesn't quite fit anywhere else.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
import pulser
from pulser import Pulse, Register
from pulser.devices import Device
from qek.shared.error import CompilationError
import rdkit.Chem as Chem


def graph_to_mol(
    graph: nx.Graph,
    node_mapping: dict[int, str],
    edge_mapping: dict[int, Chem.BondType],
) -> Chem.Mol:
    """Reconstruct an rdkit mol object using a graph.

    Args:
        graph (nx.Graph): Networkx graph of a molecule.
        mapping (MolMapping): Object containing dicts for edges and nodes
            attributes.

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


def make_sequence(device: Device, pulse: Pulse, register: Register) -> pulser.Sequence:
    """
    Build a sequence for a device from a pulse and a register.

    This function is mostly intended for internal use and will likely move to qool-layer
    in time.

    Arguments:
        device: The quantum device for which the sequence is built. Used to detect if
            a pulse + register is not compatible with a device.
        pulse: The laser pulse to apply. It will be added as a Rydberg global channel.
        register: The geometry for the sequence. If the device expects an automatic
            layout, this must already have been normalized with `with_automatic_layout`.

    Raises:
        CompilationError if the pulse + register are not compatible with the device.
    """
    try:
        sequence = pulser.Sequence(register=register, device=device)
        sequence.declare_channel("ising", "rydberg_global")
        sequence.add(pulse, "ising")
        return sequence
    except ValueError as e:
        raise CompilationError(f"This pulse/register cannot be executed on the device: {e}")
