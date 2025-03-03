from typing import cast

import os
import pulser as pl
import pytest
import torch_geometric.data as pyg_data
import torch_geometric.datasets as pyg_dataset
from qek.target import ir
from qek.target.backends import CompilationError, QutipBackend, BaseBackend
import qek.data.graphs as qek_graphs
from qek.shared.retrier import PygRetrier

if os.name == "posix":
    # As of this writing, emu-mps only works under Unix.
    from qek.target.backends import EmuMPSBackend


@pytest.mark.asyncio
async def test_async_emulators() -> None:
    """
    Test that backends based on emulators can execute without exploding (async).
    """

    # Load dataset
    original_ptcfm_data = [
        cast(pyg_data.Data, d)
        for d in PygRetrier().insist(
            pyg_dataset.TUDataset, root="dataset/test_async_emulators", name="PTC_FM"
        )
    ]

    compiled: list[tuple[qek_graphs.BaseGraph, ir.Register, ir.Pulse]] = []
    for i, data in enumerate(original_ptcfm_data):
        graph = qek_graphs.PTCFMGraph(data=data, device=pl.AnalogDevice, id=i)
        try:
            register = graph.compile_register()
            pulse = graph.compile_pulse()
            if len(register) >= 5:
                # This will be too slow to execute, skip.
                continue
        except CompilationError:
            # Let's just skip graphs that cannot be computed.
            continue
        compiled.append((graph, register, pulse))
        if len(compiled) >= 5:
            # We only need a few samples.
            break

    assert len(compiled) >= 5

    backends: list[BaseBackend] = [QutipBackend(pl.AnalogDevice)]
    if os.name == "posix":
        backends.append(EmuMPSBackend(pl.AnalogDevice))

    for backend in backends:
        for g, register, pulse in compiled:
            result = await backend.run(register, pulse)
            # The only thing we can test from the result is that the values make _some_ kind of sense.
            assert isinstance(result, dict)
            for k, v in result.items():
                assert isinstance(k, str)
                assert v >= 0
                for c in k:
                    assert c in {"0", "1"}
