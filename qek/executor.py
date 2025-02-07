import abc
from math import ceil
from typing import Counter

import os
from pulser import Pulse, Register, Sequence
from pulser.devices import Device
from pulser_simulation import QutipEmulator


class BaseExecutor(abc.ABC):
    """
    Low-level abstraction to execute a Register and a Pulse on a Quantum Device.

    For higher-level abstractions, see `BaseExtractor` and its subclasses.

    The sole role of these abstractions is to provide the same API for all backends.
    They might be removed in a future version, once Pulser has gained a similar API.
    """

    def __init__(self, device: Device):
        self.device = device

    def _make_sequence(self, register: Register, pulse: Pulse) -> Sequence:
        sequence = Sequence(register=register, device=self.device)
        sequence.declare_channel("ising", "rydberg_global")
        sequence.add(pulse, "ising")
        return sequence

    @abc.abstractmethod
    async def execute(self, register: Register, pulse: Pulse) -> dict[str, int]:
        """
        Execute a register and a pulse.

        Returns:
            A bitstring counter, i.e. a data structure counting for each bitstring
            the number of instances of this bitstring observed at the end of runs.
        """
        raise Exception("Not implemented")


class QutipExecutor(BaseExecutor):
    """
    Execute a Register and a Pulse on the Qutip Emulator.

    Please consider using EmuMPSExecutor, which generally works much better with
    higher number of qubits.

    Performance warning:
        Executing anything quantum related on an emulator takes an amount of resources
        polynomial in 2^N, where N is the number of qubits. This can easily go beyond
        the limit of the computer on which you're executing it.
    """

    def __init__(self, device: Device):
        super().__init__(device)

    async def execute(self, register: Register, pulse: Pulse) -> dict[str, int]:
        sequence = self._make_sequence(register=register, pulse=pulse)
        emulator = QutipEmulator.from_sequence(sequence)
        result: Counter[str] = emulator.run().sample_final_state()
        return result

class RemoteQPUExecutor(BaseExecutor):
    def __init__(self,
        project_id: str,
        username: str,
        device_name: str = "FRESNEL",
        password: str | None = None,
    ):
        self.project_id = project_id
        self.username = username
        self.device_name = device_name
        self.password = password


if os.name == 'posix':
    import emu_mps

    class EmuMPSExecutor(BaseExecutor):
        """
        Execute a Register and a Pulse on the high-performance emu-mps Emulator.

        Only available locally under Unix.

        Performance warning:
            Executing anything quantum related on an emulator takes an amount of resources
            polynomial in 2^N, where N is the number of qubits. This can easily go beyond
            the limit of the computer on which you're executing it.
        """

        def __init__(self, device: Device):
            super().__init__(device)

        async def execute(self, register: Register, pulse: Pulse, dt: int = 10) -> dict[str, int]:
            sequence = self._make_sequence(register=register, pulse=pulse)
            backend = emu_mps.MPSBackend()

            # Configure observable.
            cutoff_duration = int(ceil(sequence.get_duration() / dt) * dt)
            observable = emu_mps.BitStrings(evaluation_times={cutoff_duration})
            config = emu_mps.MPSConfig(observables=[observable], dt=dt)
            counter: dict[str, int] = backend.run(sequence, config)[observable.name][cutoff_duration]
            return counter

