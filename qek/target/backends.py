"""
Low-level tools to execute compiled registers and pulses onto Quantum Devices, including local emulators, remote emulators and physical QPUs.
"""

import abc
import asyncio
from typing import Counter, cast

import os
from emu_base import BackendConfig
from pasqal_cloud.device import DeviceTypeName
from pulser import Sequence
from pulser.devices import Device
from pulser.backend.remote import RemoteConnection, BatchStatus, RemoteResults, RemoteBackend
from pulser_pasqal import PasqalCloud
from pulser_simulation import QutipEmulator

from qek.data.extractors import deserialize_device
from qek.shared.error import CompilationError
from qek.shared._utils import make_sequence
from qek.target import targets


class BaseBackend(abc.ABC):
    """
    Low-level abstraction to execute a Register and a Pulse on a Quantum Device.

    For higher-level abstractions, see `BaseExtractor` and its subclasses.

    The sole role of these abstractions is to provide the same API for all backends.
    They might be removed in a future version, once Pulser has gained a similar API.
    """

    def __init__(self, device: Device | None):
        self._device = device

    def _make_sequence(self, register: targets.Register, pulse: targets.Pulse) -> Sequence:
        assert self._device is not None
        return make_sequence(register=register, pulse=pulse, device=self._device)

    @abc.abstractmethod
    async def run(self, register: targets.Register, pulse: targets.Pulse) -> Counter[str]:
        """
        Execute a register and a pulse.

        Returns:
            A bitstring Counter, i.e. a data structure counting for each bitstring
            the number of measured instances of this bitstring.
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

    async def run(self, register: targets.Register, pulse: targets.Pulse) -> Counter[str]:
        """
        Execute a register and a pulse.

        Arguments:
            register: The register (geometry) to execute. Typically obtained by compiling a graph.
            pulse: The pulse (lasers) to execute. Typically obtained by compiling a graph.

        Returns:
            A bitstring Counter, i.e. a data structure counting for each bitstring
            the number of instances of this bitstring observed at the end of runs.
        """
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
        project_id: str = None,
        username: str = None,
        device_name: str = "FRESNEL",
        password: str | None = None,
        connection: RemoteConnection = None,
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
        # validate None stuff
        if connection is not None:
            self._connection = connection
        else:
            assert project_id is not None and username is not None
            self._connection = PasqalCloud(username=username, project_id=project_id, password=password)
        self.device_name = device_name
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
        specs = self._connection.fetch_available_devices()
        self._device = cast(Device, deserialize_device(specs[self.device_name]))

        # As of this writing, the API doesn't support runs longer than 500 jobs.
        # If we want to add more runs, we'll need to split them across several jobs.
        if isinstance(self._device.max_runs, int):
            self._max_runs = self._device.max_runs

        return self._device

    async def _run(
        self,
        register: targets.Register,
        pulse: targets.Pulse,
        device_type_name: DeviceTypeName | None,
        config: BackendConfig | None = None,
        sleep_sec: int = 2,
    ) -> RemoteResults:
        """
        Run the pulse + register.

        Arguments:
            register: A register to run.
            pulse: A pulse to execute.
            device_type_name: The emulator to use, or None to run on a QPU.
            config: The backend-specific config.
            sleep_sec (optional): The amount of time to sleep when waiting for the remote server to respond, in seconds. Defaults to 2.

        Raises:
            CompilationError: If the register/pulse may not be executed on this device.
        """
        device = await self.device()
        try:
            sequence = make_sequence(device=device, pulse=pulse, register=register)

            self._sequence = sequence
        except ValueError as e:
            raise CompilationError(f"This register/pulse cannot be executed on the device: {e}")

        # Enqueue execution.
        class CloudBackend(RemoteBackend):
            def __init__(self, device_type_name, *args, **kwargs):
                self._device_type_name = device_type_name
                super(*args, **kwargs)

            def _submit_kwargs(self) -> dict[str, Any]:
                """Keyword arguments given to any call to RemoteConnection.submit()."""
                return dict(batch_id=self._batch_id, device_type=self.device_type_name)


        backend = CloudBackend(device_type_name, sequence, self._connection, config=config)
        remote_results = backend.run(
            jobs_params=[{"runs": self._max_runs}],
            wait=False,
        )

        # Wait for execution to complete.
        while True:
            await asyncio.sleep(sleep_sec)
            # Currently sync, hopefully async in the future.
            if remote_results.get_batch_status() in {BatchStatus.PENDING, BatchStatus.RUNNING}:
                # Continue waiting.
                continue
            return remote_results.results.final_bitstrings


class RemoteQPUBackend(BaseRemoteBackend):
    """
    Execute on a remote QPU.

    Performance note:
        As of this writing, the waiting lines for a QPU
        may be very long. You may use this Extractor to resume your workflow
        with a computation that has been previously started.
    """
    async def run(self, register: targets.Register, pulse: targets.Pulse) -> Counter[str]:
        remote_results = await self._run(register, pulse, device_type_name=None, config=None)
        return remote_results.results.final_bitstrings


class RemoteEmuMPSBackend(BaseRemoteBackend):
    """
    A backend that uses a remote high-performance emulator (EmuMPS)
    published on Pasqal Cloud.
    """
    async def run(self, register: targets.Register, pulse: targets.Pulse) -> Counter[str]:
        observable = emu_mps.BitStrings(evaluation_times=[1.0])
        config = emu_mps.MPSConfig(observables=[observable], dt=10)
        remote_results = await self._run(register, pulse, device_type_name=DeviceTypeName.EMU_MPS, config=config)
        return remote_results.results.final_bitstrings


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

        async def run(
            self, register: targets.Register, pulse: targets.Pulse, dt: int = 10
        ) -> Counter[str]:
            sequence = self._make_sequence(register=register, pulse=pulse)

            # Configure observable.
            observable = emu_mps.BitStrings(evaluation_times=[1.0])
            config = emu_mps.MPSConfig(observables=[observable], dt=dt)
            backend = emu_mps.MPSBackend(sequence=sequence, config=config)
            counter: Counter[str] = backend.run().get_result(observable=observable, time=1.0)
            return counter
