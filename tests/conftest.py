from __future__ import annotations

from pytest import fixture

from typing import Final

import torch_geometric.datasets as pyg_dataset
from tqdm.autonotebook import tqdm
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


@fixture
def original_ptcfm_data() -> Any:
    # Loading of the original PTC-FM dataset
    original_ptcfm_data = pyg_dataset.TUDataset(root="dataset", name="PTC_FM")
    return tqdm(original_ptcfm_data)



@fixture
def ptcfm_maps() -> tuple[dict, dict]:
    return (PTCFM_NODES_MAP, PTCFM_EDGES_MAP)
