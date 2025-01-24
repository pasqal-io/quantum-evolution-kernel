from os import path
from typing import cast

import pytest
import tempfile
import torch_geometric.data as pyg_data
import torch_geometric.datasets as pyg_dataset
import qek.data.dataset as qek_dataset
from qek.data.extractors import QutipExtractor, EmuMPSExtractor
from qek.data.graphs import MoleculeGraphCompiler


@pytest.mark.asyncio
async def test_emulators() -> None:
    """
    Test that emulators can execute without exploding.
    """
    # Load dataset
    original_ptcfm_data = [
        cast(pyg_data.Data, d) for d in pyg_dataset.TUDataset(root="dataset", name="PTC_FM")
    ]
    MAX_NUMBER_OF_SAMPLES = 5

    molecule_compiler = MoleculeGraphCompiler()

    # Test QutipExtractor
    qutip_path = path.join(tempfile.gettempdir(), "test_qutip_extractor.json")
    qutip_extractor = QutipExtractor(path=qutip_path, compiler=molecule_compiler)
    qutip_extractor.add_graphs(original_ptcfm_data[0:MAX_NUMBER_OF_SAMPLES])
    qutip_compiled = qutip_extractor.compile()
    assert (
        len(qutip_compiled) > 0
    )  # We know that some (but not all) of these samples can be compiled.
    assert len(qutip_compiled) <= MAX_NUMBER_OF_SAMPLES

    qutip_results = await qutip_extractor.run()
    assert (
        len(qutip_compiled) > 0
    )  # We know that some (but not all) of these these samples can be executed.
    assert len(qutip_results) <= len(qutip_compiled)

    # We cannot easily compare instances of `ProcessedData`, so we'll just check that deserialization doesn't explode
    # and that we obtain the expected number of items.
    qutip_loaded = qek_dataset.load_dataset(qutip_path)
    assert len(qutip_loaded) == len(qutip_results)

    # Test EmuMPSExtractor
    emu_mps_path = path.join(tempfile.gettempdir(), "test_qutip_extractor.json")
    emu_mps_extractor = EmuMPSExtractor(path=emu_mps_path, compiler=molecule_compiler)
    emu_mps_extractor.add_graphs(original_ptcfm_data[0:MAX_NUMBER_OF_SAMPLES])
    emu_mps_compiled = emu_mps_extractor.compile()
    assert len(emu_mps_compiled) > 0  # We know that these samples can be compiled.
    assert len(emu_mps_compiled) <= MAX_NUMBER_OF_SAMPLES

    emu_mps_results = await emu_mps_extractor.run()
    assert (
        len(emu_mps_results) > 0
    )  # We know that some (but not all) of these these samples can be executed.
    assert len(emu_mps_results) <= len(emu_mps_results)

    # We cannot easily compare instances of `ProcessedData`, so we'll just check that deserialization doesn't explode
    # and that we obtain the expected number of items.
    emu_mps_loaded = qek_dataset.load_dataset(emu_mps_path)
    assert len(emu_mps_loaded) == len(emu_mps_results)
    assert False
