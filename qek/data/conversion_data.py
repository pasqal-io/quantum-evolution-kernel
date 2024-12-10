from __future__ import annotations

from typing import Final

import rdkit.Chem as Chem

PTCFM_NODES_MAP: Final[dict[int, str]] = {
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

PTCFM_EDGES_MAP: Final[dict[int, Chem.BondType]] = {
    0: Chem.BondType.TRIPLE,
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.AROMATIC,
}
