"""
Low-level tools to execute compiled registers and pulses onto Quantum Devices, including local emulators, remote emulators and physical QPUs.
"""

import abc
import asyncio
from math import ceil
from typing import Counter, cast

import os
from pasqal_cloud import SDK
from pasqal_cloud.device import BaseConfig, EmulatorType
from pasqal_cloud.job import Job
from pulser import Pulse, Register, Sequence
from pulser.devices import Device
from pulser_simulation import QutipEmulator

from qek.data.extractors import deserialize_device
from qek.shared.error import CompilationError
from qek.shared._utils import make_sequence


class BaseBackend(abc.ABC):
    """
    Low-level abstraction to execute a Register and a Pulse on a Quantum Device.

    For higher-level abstractions, see `BaseExtractor` and its subclasses.

    The sole role of these abstractions is to provide the same API for all backends.
    They might be removed in a future version, once Pulser has gained a similar API.
    """

    def __init__(self, device: Device | None):
        self._device = device

    def _make_sequence(self, register: Register, pulse: Pulse) -> Sequence:
        assert self._device is not None
        return make_sequence(register=register, pulse=pulse, device=self._device)

    @abc.abstractmethod
    async def run(self, register: Register, pulse: Pulse) -> dict[str, int]:
        """
        Execute a register and a pulse.

        Returns:
            A bitstring counter, i.e. a data structure counting for each bitstring
            the number of instances of this bitstring observed at the end of runs.
        """
        raise NotImplementedError


class QutipBackend(BaseBackend):
    """
    Execute a Register and a Pulse on the Qutip Emulator.

    Please consider using EmuMPSBackend, which generally works much better with
    higher number of qubits.

    Performance warning:
        Executing anything quantum related on an emulator takes an amount of resources
        polynomial in 2^N, where N is the number of qubits. This can easily go beyond
        the limit of the computer on which you're executing it.
    """

    def __init__(self, device: Device):
        super().__init__(device)

    async def run(self, register: Register, pulse: Pulse) -> dict[str, int]:
        sequence = self._make_sequence(register=register, pulse=pulse)
        emulator = QutipEmulator.from_sequence(sequence)
        result: Counter[str] = emulator.run().sample_final_state()
        return result


class BaseRemoteBackend(BaseBackend):
    """
    Base hierarch for remote backends.

    Performance warning:
        As of this writing, using remote Backends to access a remote QPU or remote emulator
        is slower than using a RemoteExtractor, as the RemoteExtractor optimizes the number
        of connections used to communicate with the cloud server.
    """

    def __init__(
        self,
        project_id: str,
        username: str,
        device_name: str = "FRESNEL",
        password: str | None = None,
    ):
        """
        Create a remote backend

        Args:
            project_id: The ID of the project on the Pasqal Cloud API.
            username: Your username on the Pasqal Cloud API.
            password: Your password on the Pasqal Cloud API. If you leave
                this to None, you will need to enter your password manually.
            device_name: The name of the device to use. As of this writing,
                the default value of "FRESNEL" represents the latest QPU
                available through the Pasqal Cloud API.
        """
        self.device_name = device_name
        self._sdk = SDK(username=username, project_id=project_id, password=password)
        self._max_runs = 500
        self._sequence = None
        self._device = None

    async def device(self) -> Device:
        """
        Make sure that we have fetched the latest specs for the device from the server.
        """
        if self._device is not None:
            return self._device

        # Fetch the latest list of QPUs
        # Implementation note: Currently sync, hopefully async in the future.
        specs = self._sdk.get_device_specs_dict()
        self._device = cast(Device, deserialize_device(specs[self.device_name]))

        # As of this writing, the API doesn't support runs longer than 500 jobs.
        # If we want to add more runs, we'll need to split them across several jobs.
        if isinstance(self._device.max_runs, int):
            self._max_runs = self._device.max_runs

        return self._device

    async def _run(
        self,
        register: Register,
        pulse: Pulse,
        emulator: EmulatorType | None,
        config: BaseConfig | None = None,
        sleep_sec: int = 2,
    ) -> Job:
        """
        Run the pulse + register.

        Arguments:
            register: A register to run.
            pulse: A pulse to execute.
            emulator: The emulator to use, or None to run on a QPU.
            config: The backend-specific config.
            sleep_sec (optional): The amount of time to sleep when waiting for the remote server to respond, in seconds. Defaults to 2.

        Raises:
            CompilationError: If the register/pulse may not be executed on this device.
        """
        device = await self.device()
        try:
            sequence = Sequence(register=register, device=device)
            sequence.declare_channel("ising", "rydberg_global")
            sequence.add(pulse=pulse, channel="ising")

            self._sequence = sequence
        except ValueError as e:
            raise CompilationError(f"This register/pulse cannot be executed on the device: {e}")

        # Enqueue execution.
        batch = self._sdk.create_batch(
            serialized_sequence=sequence.to_abstract_repr(),
            jobs=[{"runs": self._max_runs}],
            wait=False,
            emulator=emulator,
            configuration=config,
        )

        # Wait for execution to complete.
        while True:
            await asyncio.sleep(sleep_sec)
            # Currently sync, hopefully async in the future.
            batch.refresh()
            if batch.status in {"PENDING", "RUNNING"}:
                # Continue waiting.
                continue
            job = next(iter(batch.jobs.values()))
            if job.status == "ERROR":
                raise Exception(f"Error while executing remote job: {job.errors}")
            return job


class RemoteQPUBackend(BaseRemoteBackend):
    """
    Execute on a remote QPU.

    Performance note:
        As of this writing, the waiting lines for a QPU
        may be very long. You may use this Extractor to resume your workflow
        with a computation that has been previously started.
    """

    async def run(self, register: Register, pulse: Pulse) -> dict[str, int]:
        job = await self._run(register, pulse, emulator=None, config=None)
        return cast(dict[str, int], job.result)


class RemoteEmuMPSBackend(BaseRemoteBackend):
    """
    A backend that uses a remote high-performance emulator (EmuMPS)
    published on Pasqal Cloud.
    """

    async def run(self, register: Register, pulse: Pulse, dt: int = 10) -> dict[str, int]:
        job = await self._run(register, pulse, emulator=EmulatorType.EMU_MPS, config=None)
        bag = cast(dict[str, dict[int, dict[str, int]]], job.result)

        assert self._sequence is not None
        cutoff_duration = int(ceil(self._sequence.get_duration() / dt) * dt)
        return bag["bitstring"][cutoff_duration]


if os.name == "posix":
    import emu_mps

    class EmuMPSBackend(BaseBackend):
        """
        Execute a Register and a Pulse on the high-performance emu-mps Emulator.

        As of this writing, this local emulator is only available under Unix. However,
        the RemoteEmuMPSBackend is available on all platforms.

        Performance warning:
            Executing anything quantum related on an emulator takes an amount of resources
            polynomial in 2^N, where N is the number of qubits. This can easily go beyond
            the limit of the computer on which you're executing it.
        """

        def __init__(self, device: Device):
            super().__init__(device)

        async def run(self, register: Register, pulse: Pulse, dt: int = 10) -> dict[str, int]:
            sequence = self._make_sequence(register=register, pulse=pulse)
            backend = emu_mps.MPSBackend()

            # Configure observable.
            cutoff_duration = int(ceil(sequence.get_duration() / dt) * dt)
            observable = emu_mps.BitStrings(evaluation_times={cutoff_duration})
            config = emu_mps.MPSConfig(observables=[observable], dt=dt)
            counter: dict[str, int] = backend.run(sequence, config)[observable.name][
                cutoff_duration
            ]
            return counter
