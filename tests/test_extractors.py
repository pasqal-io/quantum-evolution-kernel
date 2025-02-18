from typing import cast

import os
import pytest
import torch_geometric.data as pyg_data
import torch_geometric.datasets as pyg_dataset
import conftest
from qek.data.extractors import QutipExtractor, BaseExtracted

if os.name == "posix":
    # As of this writing, emu-mps only works under Unix.
    from qek.data.extractors import EmuMPSExtractor

from qek.data.graphs import PTCFMCompiler


@pytest.mark.asyncio
async def test_async_emulators() -> None:
    """
    Test that extractors emulators can execute without exploding (both sync and async).
    """
    conftest.preload_dataset()

    # Load dataset
    original_ptcfm_data = [
        cast(pyg_data.Data, d) for d in pyg_dataset.TUDataset(root="dataset", name="PTC_FM")
    ]
    MAX_NUMBER_OF_SAMPLES = 5
    MAX_NUMBER_OF_QUBITS = 5

    ptcfm_compiler = PTCFMCompiler()

    all_qutip_results: dict[bool, BaseExtracted] = {}
    all_emu_mps_results: dict[bool, BaseExtracted] = {}
    for sync in [True, False]:
        # Test QutipExtractor
        qutip_extractor = QutipExtractor(compiler=ptcfm_compiler)
        qutip_extractor.add_graphs(original_ptcfm_data[0:MAX_NUMBER_OF_SAMPLES])
        qutip_compiled = qutip_extractor.compile()
        assert (
            len(qutip_compiled) > 0
        )  # We know that some (but not all) of these samples can be compiled.
        assert len(qutip_compiled) <= MAX_NUMBER_OF_SAMPLES

        qutip_results = qutip_extractor.run(max_qubits=MAX_NUMBER_OF_QUBITS)
        all_qutip_results[sync] = qutip_results
        if not sync:
            await qutip_results
        assert (
            len(qutip_compiled) > 0
        )  # We know that some (but not all) of these these samples can be executed.
        assert len(qutip_results.raw_data) <= len(qutip_compiled)

        # Test EmuMPSExtractor
        if os.name == "posix":
            emu_mps_extractor = EmuMPSExtractor(compiler=ptcfm_compiler)
            emu_mps_extractor.add_graphs(original_ptcfm_data[0:MAX_NUMBER_OF_SAMPLES])
            emu_mps_compiled = emu_mps_extractor.compile()
            assert len(emu_mps_compiled) > 0  # We know that these samples can be compiled.
            assert len(emu_mps_compiled) <= MAX_NUMBER_OF_SAMPLES

            emu_mps_results = emu_mps_extractor.run(max_qubits=MAX_NUMBER_OF_QUBITS)
            all_emu_mps_results[sync] = emu_mps_results
            if not sync:
                await emu_mps_results
            assert (
                len(emu_mps_results.raw_data) > 0
            )  # We know that some (but not all) of these these samples can be executed.
            assert len(emu_mps_results.raw_data) <= len(emu_mps_results.raw_data)

            # Compare what we can, which isn't much.
            assert emu_mps_results.targets == qutip_results.targets

    assert all_qutip_results[True].targets == all_qutip_results[False].targets
    assert len(all_qutip_results[True].processed_data) == len(
        all_qutip_results[False].processed_data
    )

    if os.name == "posix":
        assert all_emu_mps_results[True].targets == all_emu_mps_results[False].targets
        assert len(all_emu_mps_results[True].processed_data) == len(
            all_emu_mps_results[False].processed_data
        )
