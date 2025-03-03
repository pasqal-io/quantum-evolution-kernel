"""
High-Level API to compile raw data (graphs) and process it on a quantum device, either a local emulator,
a remote emulator or a physical QPI.
"""

import abc
import asyncio
from dataclasses import dataclass
import itertools
import json
import logging
from math import ceil
from uuid import UUID
import time
from typing import Any, Callable, Generator, Generic, Sequence, TypeVar, cast
from numpy.typing import NDArray
from pasqal_cloud import SDK
from pasqal_cloud.batch import Batch
from pasqal_cloud.device import BaseConfig, EmuTNConfig, EmulatorType
from pasqal_cloud.job import Job
from pasqal_cloud.utils.filters import BatchFilters
from pathlib import Path
import numpy as np
import os
import pulser as pl
from pulser.devices import Device
from pulser.json.abstract_repr.deserializer import deserialize_device
from pulser_simulation import QutipEmulator
from torch.utils.data import Dataset

from qek.data import processed_data
from qek.data.graphs import BaseGraph, BaseGraphCompiler
from qek.data.processed_data import ProcessedData
from qek.shared.error import CompilationError

logger = logging.getLogger(__name__)


@dataclass
class Compiled:
    """
    The result of compiling a graph for execution on a quantum device.
    """

    # Future plans: as of this writing, this class (or a reworked version of it)
    # is expected to move to the `qool-layer` library.

    # The graph itself.
    graph: BaseGraph

    # A sequence adapted to the quantum device.
    sequence: pl.Sequence


@dataclass
class Feature:
    """
    A feature extracted from raw data.
    """

    data: NDArray[np.floating]


class BaseExtracted(abc.ABC):
    """
    Data extracted by one of the subclasses of `BaseExtractor`.

    Note that the list of processed data will generally *not* contain all the graphs ingested
    by the Extractor, as not all graphs may not be compiled for a given device.
    """

    def __init__(self, device: Device):
        self.device = device

    def __await__(self) -> Generator[Any, Any, None]:
        """
        Wait asynchronously until execution is ready.

        This will avoid blocking your main thread, so calling this method once,
        before the first call to `processed_data`, is strongly recommended
        for use on a server or an interactive application.
        """
        # By default, no need to wait.
        yield None

    @property
    @abc.abstractmethod
    def processed_data(self) -> list[processed_data.ProcessedData]:
        pass

    @property
    @abc.abstractmethod
    def raw_data(self) -> list[BaseGraph]:
        """
        A subset of the graphs ingested by the Extractor.
        """
        pass

    @property
    @abc.abstractmethod
    def targets(self) -> list[int] | None:
        """
        If available, the machine-learning targets for these graphs, in the same order and with the same number of entrie as `raw_data`.
        """
        pass

    @property
    def states(self) -> list[dict[str, int]]:
        """
        The quantum states extracted from `raw_data` by executing `sequences` on the device, in the same order and with the same number of entries as `raw_data`.
        """
        return [data.state_dict for data in self.processed_data]

    def features(self, size_max: int | None) -> list[Feature]:
        """
        The features extracted from `raw_data` by processing `states`, in the same order and with the same number of entries as `raw_data`.

        By default, the features extracted are the distribution of excitation levels based on `states`. However, subclasses may override
        this method to provide custom features extraction.

        Arguments:
            size_max (optional) Performance/precision lever. If specified, specifies the number of qubits to take into account from all
                the `states`. If `size_max` is lower than the number of qubits used to extract `self.states[i]` (i.e. the number of qubits
                in `self.sequences[i]`), then only take into account the `size_max` first qubits of this state to extract
                `self.features(size_max)[i]`. If, on the other hand, `size_max` is greater than the number of qubits used to extract
                `self.states[i]`, pad `self.features(size_max)[i]` with 0s.
                If unspecified, use the largest number of qubits in `selfsequences`.
        """
        if size_max is None:
            for data in self.processed_data:
                seq = data._sequence
                if size_max is None or len(seq.qubit_info) > size_max:
                    size_max = len(seq.qubit_info)
        if size_max is None:
            # The only way size_max can be None is if `self.sequences` is empty.
            return []

        return [Feature(processed_data.dist_excitation(state, size_max)) for state in self.states]

    def save_dataset(self, file_path: Path) -> None:
        """Saves the processed dataset to a JSON file.

        Note: This does NOT attempt to save the graphs.

        Args:
            dataset: The dataset to be saved.
            file_path: The path where the dataset will be saved as a JSON
                file.

        Note:
            The data is stored in a format suitable for loading with load_dataset.
        """
        with open(file_path, "w") as file:
            states = self.states
            targets = self.targets
            data = [
                {
                    "sequence": self.processed_data[i]._sequence.to_abstract_repr(),
                    # Some emulators will actually be `dict[str, int64]` instead of `dict[str, int]` and `int64`
                    # is not JSON-serializable.
                    #
                    # The reason for which `int64` is not JSON-serializable is that JSON limits ints to 2^53-1.
                    # However, in practice, this should not be a problem, since the `int`/`int64` in our dict is
                    # limited to the number of runs, and we don't expect to be launching 2^53 consecutive runs
                    # for a single sequence on a device in any foreseeable future (assuming a run of 1ns,
                    # this would still take ~4 billion years to execute).
                    "state_dict": {key: int(value) for (key, value) in states[i].items()},
                    "target": targets[i] if targets is not None else None,
                }
                for i in range(len(self.processed_data))
            ]
            json.dump(data, file)
        logger.info("processed data saved to %s", file_path)


class SyncExtracted(BaseExtracted):
    """
    Data extracted synchronously, i.e. no need to wait for a remote server.
    """

    def __init__(
        self,
        raw_data: list[BaseGraph],
        targets: list[int] | None,
        sequences: list[pl.Sequence],
        states: list[dict[str, int]],
    ):
        assert len(raw_data) == len(sequences)
        assert len(sequences) == len(states)
        if targets is not None:
            if len(targets) < len(sequences):
                # Not all graphs come with a target.
                #
                # This Extracted will not be usable as the training sample, so ignore all targets.
                if len(targets) != 0:
                    logger.debug(
                        "We compiled %s graphs but we only have %s targets, ignoring all targets",
                        len(sequences),
                        len(targets),
                    )
                targets = None
        self._raw_data = raw_data
        self._targets = targets
        self._sequences = sequences
        self._states = states
        self._processed_data = [
            ProcessedData(
                sequence=seq, state_dict=cast(dict[str, int | np.int64], state), target=target
            )
            for (seq, state, target) in itertools.zip_longest(sequences, states, targets or [])
        ]

    @property
    def processed_data(self) -> list[ProcessedData]:
        return self._processed_data

    @property
    def raw_data(self) -> list[BaseGraph]:
        return self._raw_data

    @property
    def targets(self) -> list[int] | None:
        return self._targets

    @property
    def sequences(self) -> list[pl.Sequence]:
        return self._sequences

    @property
    def states(self) -> list[dict[str, int]]:
        return self._states


# Type variable for BaseExtractor[GraphType].
GraphType = TypeVar("GraphType")


class BaseExtractor(abc.ABC, Generic[GraphType]):
    """
    The base of the hierarchy of extractors.

    The role of extractors is to take a list of raw data (here, labelled graphs) into
    processed data containing machine-learning features (here, excitation vectors).

    Args:
        path: If specified, the processed data will be saved to this file as JSON once
            the execution is complete.
        device: A quantum device for which the data should be prepared.
        compiler: A graph compiler, in charge of converting graphs to Pulser Sequences,
            the format that can be executed on a quantum device.
    """

    def __init__(
        self, device: Device, compiler: BaseGraphCompiler[GraphType], path: Path | None = None
    ) -> None:
        self.path = path

        # The list of graphs (raw data). Fill it with `self.add_graphs`.
        self.graphs: list[BaseGraph] = []
        self.device: Device = device

        # The compiled sequences. Filled with `self.compile`.
        # Note that the list of compiled sequences may be shorter than the list of
        # raw data, as not all graphs may be compiled to a given `device`.
        self.sequences: list[Compiled] = []
        self.compiler = compiler

        # A counter used to give a unique id to each graph.
        self._counter = 0

    def save(self, snapshot: list[ProcessedData]) -> None:
        """Saves a dataset to a JSON file.

        Args:
            dataset (list[ProcessedData]): The dataset to be saved, containing
                RegisterData instances.
            file_path (str): The path where the dataset will be saved as a JSON
                file.

        Note:
            The data is stored in a format suitable for loading with load_dataset.
        """
        if self.path is not None:
            with open(self.path, "w") as file:
                data = [
                    {
                        "sequence": instance._sequence.to_abstract_repr(),
                        "state_dict": instance.state_dict,
                        "target": instance.target,
                    }
                    for instance in snapshot
                ]
                json.dump(data, file)
            logger.info("processed data saved to %s", self.path)

    def compile(
        self, filter: Callable[[BaseGraph, pl.Sequence, int], bool] | None = None
    ) -> list[Compiled]:
        """
        Compile all pending graphs into Pulser sequences that the Quantum Device may execute.

        Once this method has succeeded, the results are stored in `self.sequences`.
        """
        if len(self.graphs) == 0:
            raise Exception("No graphs to compile, did you forget to call `import_graphs`?")
        if filter is None:
            filter = lambda _graph, sequence, _index: True  # noqa: E731
        self.sequences = []
        for graph in self.graphs:
            try:
                register = graph.compile_register()
                pulse = graph.compile_pulse()
                sequence = pl.Sequence(register=register, device=graph.device)
                sequence.declare_channel("ising", "rydberg_global")
                sequence.add(pulse, "ising")
            except CompilationError as e:
                # In some cases, we produce graphs that pass `is_embeddable` but cannot be compiled.
                # It _looks_ like this is due to rounding errors. We're investigating this in issue #29,
                # but for the time being, we're simply logging and skipping them.
                logger.debug("Graph #%s could not be compiled (%s), skipping", graph.id, e)
                continue
            if not filter(graph, sequence, graph.id):
                logger.debug("Graph #%s did not pass filtering, skipping", graph.id)
                continue
            logger.debug("Compiling graph #%s for execution on the device", graph.id)
            self.sequences.append(Compiled(graph=graph, sequence=sequence))
        logger.debug("Compilation step complete, %s graphs compiled", len(self.sequences))
        return self.sequences

    def add_graphs(self, graphs: Sequence[GraphType] | Dataset[GraphType]) -> None:
        """
        Add new graphs to compile and run.
        """
        for graph in graphs:
            self._counter += 1
            id = self._counter
            logger.debug("ingesting # %s", id)
            processed = self.compiler.ingest(graph=graph, device=self.device, id=id)
            # Skip graphs that are not embeddable.
            if processed.is_embeddable():
                logger.debug("graph # %s is embeddable, accepting", id)
                self.graphs.append(processed)
            else:
                logger.info("graph # %s is not embeddable, skipping", id)
        logger.info("imported %s graphs", len(self.graphs))

    @abc.abstractmethod
    def run(self) -> BaseExtracted:
        """
        Run compiled graphs.

        You will need to call `self.compile` first, to make sure that the graphs are compiled.

        Returns:
            Data extracted by this extractor.

            Not all extractors may return the same data, so please take a look at the documentation
            of the extractor you are using.
        """
        raise Exception("Not implemented")


class QutipExtractor(BaseExtractor[GraphType]):
    """
    A Extractor that uses the Qutip Emulator to run sequences compiled
    from graphs.

    Performance note: emulating a quantum device on a classical
    computer requires considerable amount of resources, so this
    Extractor may be slow or require too much memory.

    See also:
    - EmuMPSExtractor (alternative emulator, generally much faster)
    - QPUExtractor (run on a physical QPU)

    Args:
        path: Path to store the result of the run, for future uses.
            To reload the result of a previous run, use `LoadExtractor`.
        compiler: A graph compiler, in charge of converting graphs to Pulser Sequences,
            the format that can be executed on a quantum device.
        device: A device to use. For general experiments, the default
            device `AnalogDevice` is a perfectly reasonable choice.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        device: Device = pl.devices.AnalogDevice,
        path: Path | None = None,
    ):
        super().__init__(path=path, device=device, compiler=compiler)
        self.graphs: list[BaseGraph]
        self.device = device

    def run(self, max_qubits: int = 8) -> SyncExtracted:
        """
        Run the compiled graphs.

        As emulating a quantum device is slow consumes resources and time exponential in the
        number of qubits, for the sake of performance, we limit the number of qubits in the execution
        of this extractor.

        Args:
            max_qubits: Skip any sequence that require strictly more than `max_qubits`. Defaults to 8.

        Returns:
            Processed data for all the sequences that were executed.
        """
        if len(self.sequences) == 0:
            logger.warning("No sequences to run, did you forget to call compile()?")
            return SyncExtracted(raw_data=[], targets=[], sequences=[], states=[])

        raw_data: list[BaseGraph] = []
        targets: list[int] = []
        sequences: list[pl.Sequence] = []
        states: list[dict[str, int]] = []
        for compiled in self.sequences:
            qubits_used = len(compiled.sequence.qubit_info)
            if qubits_used > max_qubits:
                logger.info(
                    "Graph %s exceeds the qubit limit specified in QutipExtractor (%s > %s), skipping",
                    id,
                    qubits_used,
                    max_qubits,
                )
                continue
            logger.debug("Executing compiled graph # %s", id)
            simul = QutipEmulator.from_sequence(sequence=compiled.sequence)
            counter = cast(dict[str, Any], simul.run().sample_final_state())
            logger.debug("Execution of compiled graph # %s complete", id)
            raw_data.append(compiled.graph)
            if compiled.graph.target is not None:
                targets.append(compiled.graph.target)
            sequences.append(compiled.sequence)
            states.append(counter)

        result = SyncExtracted(
            raw_data=raw_data, targets=targets, sequences=sequences, states=states
        )
        logger.debug("Emulation step complete, %s compiled graphs executed", len(raw_data))
        if self.path is not None:
            result.save_dataset(self.path)
        return result


if os.name == "posix":
    # Any Unix including Linux and macOS

    import emu_mps

    class EmuMPSExtractor(BaseExtractor[GraphType]):
        """
        A Extractor that uses the emu-mps Emulator to run sequences compiled
        from graphs.

        Performance note: emulating a quantum device on a classical
        computer requires considerable amount of resources, so this
        Extractor may be slow or require too much memory. If should,
        however, be faster than QutipExtractor in most cases.

        See also:
        - QPUExtractor (run on a physical QPU)

        Args:
            path: Path to store the result of the run, for future uses.
                To reload the result of a previous run, use `LoadExtractor`.
            compiler: A graph compiler, in charge of converting graphs to Pulser Sequences,
                the format that can be executed on a quantum device.
            device: A device to use. For general experiments, the default
                device `AnalogDevice` is a perfectly reasonable choice.
        """

        def __init__(
            self,
            compiler: BaseGraphCompiler[GraphType],
            device: Device = pl.devices.AnalogDevice,
            path: Path | None = None,
        ):
            super().__init__(device=device, compiler=compiler, path=path)
            self.graphs: list[BaseGraph]
            self.device = device

        def run(self, max_qubits: int = 10, dt: int = 10) -> BaseExtracted:
            """
            Run the compiled graphs.

            As emulating a quantum device is slow consumes resources and time exponential in the
            number of qubits, for the sake of performance, we limit the number of qubits in the execution
            of this extractor.

            Args:
                max_qubits: Skip any sequence that require strictly more than `max_qubits`. Defaults to 8.
                dt: The duration of the simulation step, in us. Defaults to 10.

            Returns:
                Processed data for all the sequences that were executed.
            """
            if len(self.sequences) == 0:
                logger.warning("No sequences to run, did you forget to call compile()?")
                return SyncExtracted(raw_data=[], targets=[], sequences=[], states=[])

            backend = emu_mps.MPSBackend()
            raw_data = []
            targets: list[int] = []
            sequences = []
            states = []
            for compiled in self.sequences:
                qubits_used = len(compiled.sequence.qubit_info)
                if qubits_used > max_qubits:
                    logger.info(
                        "Graph %s exceeds the qubit limit specified in EmuMPSExtractor (%s > %s), skipping",
                        id,
                        qubits_used,
                        max_qubits,
                    )
                    continue
                logger.debug("Executing compiled graph # %s", id)

                # Configure observable.
                cutoff_duration = int(ceil(compiled.sequence.get_duration() / dt) * dt)
                observable = emu_mps.BitStrings(evaluation_times={cutoff_duration})
                config = emu_mps.MPSConfig(observables=[observable], dt=dt)
                counter: dict[str, Any] = backend.run(compiled.sequence, config)[observable.name][
                    cutoff_duration
                ]
                logger.debug("Execution of compiled graph # %s complete", id)
                raw_data.append(compiled.graph)
                if compiled.graph.target is not None:
                    targets.append(compiled.graph.target)
                sequences.append(compiled.sequence)
                states.append(counter)

            logger.debug("Emulation step complete, %s compiled graphs executed", len(raw_data))

            result = SyncExtracted(
                raw_data=raw_data, targets=targets, sequences=sequences, states=states
            )
            logger.debug("Emulation step complete, %s compiled graphs executed", len(raw_data))
            if self.path is not None:
                result.save_dataset(self.path)
            return result


# How many seconds to sleep while waiting for the results from the cloud.
SLEEP_DELAY_S = 2


class PasqalCloudExtracted(BaseExtracted):
    """
    Data extracted from the cloud API, i.e. we need wait for a remote server.

    Performance note:
        If your code is meant to be executed as part of an interactive application or
        a server, you should consider calling `await extracted` before your first call
        to any of the methods of `extracted`. Otherwise, you will block the main thread.

        If you are running this as part of an experiment, a Jupyter notebook, etc. you
        do not need to do so.
    """

    def __init__(
        self,
        compiled: list[Compiled],
        batch_ids: list[str],
        sdk: SDK,
        state_extractor: Callable[[Job, pl.Sequence], dict[str, int] | None],
        path: Path | None = None,
    ):
        """
        Prepare for reception of data.

        Arguments:
            compiled: The result of compiling a set of graphs.
            batch_ids: The ids of the batches on the cloud API, in the same order as `compiled`.
            state_extractor: A callback used to extract the counter from a job.
                Used as various cloud back-ends return different formats.
            path: If provided, a path at which to save the results once they're available.
        """
        self._compiled = compiled
        self._batch_ids = batch_ids
        self._results: SyncExtracted | None = None
        self._path = path
        self._sdk = sdk
        self._state_extractor = state_extractor

    def _wait(self) -> None:
        """
        Wait synchronously until remote execution is ready.

        This WILL BLOCK your main thread, possibly for a very long time.
        """
        if self._results is not None:
            # Results are already available.
            return
        pending_batch_ids: set[str] = set(self._batch_ids)
        completed_batches: dict[str, Batch] = {}
        while len(pending_batch_ids) > 0:
            time.sleep(SLEEP_DELAY_S)

            # Fetch up to 100 pending batches (upstream limits).
            MAX_BATCH_LEN = 100
            check_ids: list[str | UUID] = [cast(str | UUID, id) for id in pending_batch_ids][
                :MAX_BATCH_LEN
            ]

            # Update their status.
            check_batches = self._sdk.get_batches(filters=BatchFilters(id=check_ids))
            for batch in check_batches.results:
                assert isinstance(batch, Batch)
                if batch.status not in {"PENDING", "RUNNING"}:
                    logger.debug("Job %s is now complete", batch.id)
                    pending_batch_ids.discard(batch.id)
                    completed_batches[batch.id] = batch

        # At this point, all batches are complete.
        self._ingest(completed_batches)

    def __await__(self) -> Generator[Any, Any, None]:
        """
        Wait asynchronously until remote execution is ready.

        This will NOT block your main thread, so this method is strongly recommended
        for use on a server or an interactive application.

        Example:
            await extracted
        """
        if self._results is not None:
            # Results are already available.
            return
        pending_batch_ids: set[str] = set(self._batch_ids)
        completed_batches: dict[str, Batch] = {}
        while len(pending_batch_ids) > 0:
            yield from asyncio.sleep(SLEEP_DELAY_S).__await__()

            # Fetch up to 100 pending batches (upstream limits).
            MAX_BATCH_LEN = 100
            check_ids: list[str | UUID] = [cast(str | UUID, id) for id in pending_batch_ids][
                :MAX_BATCH_LEN
            ]

            # Update their status.
            check_batches = self._sdk.get_batches(
                filters=BatchFilters(id=check_ids)
            )  # Ideally, this should be async, see https://github.com/pasqal-io/pasqal-cloud/issues/162.
            for batch in check_batches.results:
                assert isinstance(batch, Batch)
                if batch.status not in {"PENDING", "RUNNING"}:
                    logger.debug("Job %s is now complete", batch.id)
                    pending_batch_ids.discard(batch.id)
                    completed_batches[batch.id] = batch

        # At this point, all batches are complete.
        self._ingest(completed_batches)

    def _ingest(self, batches: dict[str, Batch]) -> None:
        """
        Ingest data received from the remote server.

        No I/O.
        """
        assert len(batches) == len(self._batch_ids)

        raw_data = []
        targets: list[int] = []
        sequences = []
        states = []
        for i, id in enumerate(self._batch_ids):
            batch = batches[id]
            compiled = self._compiled[i]
            # Note: There's only one job per batch.
            assert len(batch.jobs) == 1
            for job in batch.jobs.values():
                if job.status == "DONE":
                    state_dict = self._state_extractor(job, compiled.sequence)
                    if state_dict is None:
                        logger.warning(
                            "Batch %s (graph %s) did not return a usable state, skipping",
                            i,
                            compiled.graph.id,
                        )
                        continue
                    raw_data.append(compiled.graph)
                    if compiled.graph.target is not None:
                        targets.append(compiled.graph.target)
                    sequences.append(compiled.sequence)
                    states.append(state_dict)
                else:
                    # If some sequences failed, let's skip them and proceed as well as we can.
                    logger.warning(
                        "Batch %s (graph %s) failed with errors %s, skipping",
                        i,
                        compiled.graph.id,
                        job.status,
                        job.errors,
                    )
        self._results = SyncExtracted(
            raw_data=raw_data, targets=targets, sequences=sequences, states=states
        )
        if self._path is not None:
            self.save_dataset(self._path)

    @property
    def processed_data(self) -> list[ProcessedData]:
        self._wait()
        assert self._results is not None
        return self._results.processed_data

    @property
    def raw_data(self) -> list[BaseGraph]:
        self._wait()
        assert self._results is not None
        return self._results.raw_data

    @property
    def targets(self) -> list[int] | None:
        self._wait()
        assert self._results is not None
        return self._results.targets

    @property
    def sequences(self) -> list[pl.Sequence]:
        self._wait()
        assert self._results is not None
        return self._results.sequences

    @property
    def states(self) -> list[dict[str, int]]:
        self._wait()
        assert self._results is not None
        return self._results.states


class BaseRemoteExtractor(BaseExtractor[GraphType], Generic[GraphType]):
    """
    An Extractor that uses a remote Quantum Device published
    on Pasqal Cloud, to run sequences compiled from graphs.

    Performance note (servers and interactive applications only):
        If your code is meant to be executed as part of an interactive application or
        a server, you should consider calling `await extracted` before your first call
        to any of the methods of `extracted`. Otherwise, you will block the main thread.

        If you are running this as part of an experiment, a Jupyter notebook, etc. you
        may ignore this performance note.

    Args:
        path: Path to store the result of the run, for future uses.
            To reload the result of a previous run, use `LoadExtractor`.
        project_id: The ID of the project on the Pasqal Cloud API.
        username: Your username on the Pasqal Cloud API.
        password: Your password on the Pasqal Cloud API. If you leave
            this to None, you will need to enter your password manually.
        device_name: The name of the device to use. As of this writing,
            the default value of "FRESNEL" represents the latest QPU
            available through the Pasqal Cloud API.
        batch_id: Use this to resume a workflow e.g. after turning off
            your computer while the QPU was executing your sequences.
            Warning: A batch started with one executor MUST NOT be resumed
            with a different executor.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        project_id: str,
        username: str,
        device_name: str,
        password: str | None = None,
        batch_ids: list[str] | None = None,
        path: Path | None = None,
    ):
        sdk = SDK(username=username, project_id=project_id, password=password)

        # Fetch the latest list of QPUs
        specs = sdk.get_device_specs_dict()
        device = cast(Device, deserialize_device(specs[device_name]))

        super().__init__(device=device, compiler=compiler, path=path)
        self._sdk = sdk
        self._batch_ids: list[str] | None = batch_ids

    @property
    def batch_ids(self) -> list[str] | None:
        return self._batch_ids

    @abc.abstractmethod
    def run(
        self,
    ) -> PasqalCloudExtracted:
        """
        Launch the extraction.
        """
        raise NotImplementedError()

    def _run(
        self,
        state_extractor: Callable[[Job, pl.Sequence], dict[str, int] | None],
        emulator: EmulatorType | None,
        config: BaseConfig | None,
    ) -> PasqalCloudExtracted:
        if len(self.sequences) == 0:
            logger.warning("No sequences to run, did you forget to call compile()?")
            return PasqalCloudExtracted(
                compiled=[],
                batch_ids=[],
                sdk=self._sdk,
                path=self.path,
                state_extractor=state_extractor,
            )

        device: pl.devices.Device = self.sequences[0].sequence.device
        # As of this writing, the API doesn't support runs longer than 500 jobs.
        # If we want to add more runs, we'll need to split them across several jobs.
        max_runs = device.max_runs if isinstance(device.max_runs, int) else 500

        if self._batch_ids is None:
            # Enqueue jobs.
            self._batch_ids = []
            for compiled in self.sequences:
                logger.debug("Enqueuing execution of compiled graph #%s", compiled.graph.id)
                batch = self._sdk.create_batch(
                    compiled.sequence.to_abstract_repr(),
                    jobs=[{"runs": max_runs}],
                    wait=False,
                    emulator=emulator,
                    configuration=config,
                )
                logger.info(
                    "Remote execution of compiled graph #%s starting, batched with id %s",
                    compiled.graph.id,
                    batch.id,
                )
                self._batch_ids.append(batch.id)
            logger.info(
                "All %s jobs enqueued for remote execution, with ids %s",
                len(self._batch_ids),
                self._batch_ids,
            )
        assert len(self._batch_ids) == len(self.sequences)

        return PasqalCloudExtracted(
            compiled=self.sequences,
            batch_ids=self._batch_ids,
            sdk=self._sdk,
            path=self.path,
            state_extractor=state_extractor,
        )


class RemoteQPUExtractor(BaseRemoteExtractor[GraphType]):
    """
    An Extractor that uses a remote QPU published
    on Pasqal Cloud, to run sequences compiled from graphs.

    Performance note:
        as of this writing, the waiting lines for a QPU
        may be very long. You may use this Extractor to resume your workflow
        with a computation that has been previously started.

    Performance note (servers and interactive applications only):
        If your code is meant to be executed as part of an interactive application or
        a server, you should consider calling `await extracted` before your first call
        to any of the methods of `extracted`. Otherwise, you will block the main thread.

        If you are running this as part of an experiment, a Jupyter notebook, etc. you
        may ignore this performance note.

    Args:
        path: Path to store the result of the run, for future uses.
            To reload the result of a previous run, use `LoadExtractor`.
        project_id: The ID of the project on the Pasqal Cloud API.
        username: Your username on the Pasqal Cloud API.
        password: Your password on the Pasqal Cloud API. If you leave
            this to None, you will need to enter your password manually.
        device_name: The name of the device to use. As of this writing,
            the default value of "FRESNEL" represents the latest QPU
            available through the Pasqal Cloud API.
        batch_id: Use this to resume a workflow e.g. after turning off
            your computer while the QPU was executing your sequences.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        project_id: str,
        username: str,
        device_name: str = "FRESNEL",
        password: str | None = None,
        batch_ids: list[str] | None = None,
        path: Path | None = None,
    ):
        super().__init__(
            compiler=compiler,
            project_id=project_id,
            username=username,
            device_name=device_name,
            password=password,
            batch_ids=batch_ids,
            path=path,
        )

    def run(self) -> PasqalCloudExtracted:
        return self._run(emulator=None, config=None, state_extractor=lambda job, _seq: job.result)


class RemoteEmuMPSExtractor(BaseRemoteExtractor[GraphType]):
    """
    An Extractor that uses a remote high-performance emulator (EmuMPS)
    published on Pasqal Cloud, to run sequences compiled from graphs.

    Performance note (servers and interactive applications only):
        If your code is meant to be executed as part of an interactive application or
        a server, you should consider calling `await extracted` before your first call
        to any of the methods of `extracted`. Otherwise, you will block the main thread.

        If you are running this as part of an experiment, a Jupyter notebook, etc. you
        may ignore this performance note.

    Args:
        path: Path to store the result of the run, for future uses.
            To reload the result of a previous run, use `LoadExtractor`.
        project_id: The ID of the project on the Pasqal Cloud API.
        username: Your username on the Pasqal Cloud API.
        password: Your password on the Pasqal Cloud API. If you leave
            this to None, you will need to enter your password manually.
        device_name: The name of the device to use. As of this writing,
            the default value of "FRESNEL" represents the latest QPU
            available through the Pasqal Cloud API.
        batch_id: Use this to resume a workflow e.g. after turning off
            your computer while the QPU was executing your sequences.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        project_id: str,
        username: str,
        device_name: str = "FRESNEL",
        password: str | None = None,
        batch_ids: list[str] | None = None,
        path: Path | None = None,
    ):
        super().__init__(
            compiler=compiler,
            project_id=project_id,
            username=username,
            device_name=device_name,
            password=password,
            batch_ids=batch_ids,
            path=path,
        )

    def run(self, dt: int = 10) -> PasqalCloudExtracted:
        def extractor(job: Job, sequence: pl.Sequence) -> dict[str, int] | None:
            cutoff_duration = int(ceil(sequence.get_duration() / dt) * dt)
            full_result = job.full_result
            if full_result is None:
                return None
            result = full_result["bitstring"][cutoff_duration]
            if result is None:
                return None
            assert isinstance(result, dict)
            return result

        return self._run(
            emulator=EmulatorType.EMU_MPS,
            config=EmuTNConfig(
                dt=dt,
            ),
            state_extractor=extractor,
        )
