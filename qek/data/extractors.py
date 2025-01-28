import abc
from asyncio import sleep
from dataclasses import dataclass
import json
import logging
from math import ceil
from typing import Any, Callable, Generic, Sequence, TypeVar, cast
import emu_mps
from pasqal_cloud import SDK, Batch, BatchFilters
import pulser as pl
from pulser.devices import Device
from pulser.json.abstract_repr.deserializer import deserialize_device
from pulser_simulation import QutipEmulator

from qek.data.dataset import ProcessedData
from qek.data.graphs import BaseGraph, BaseGraphCompiler

logger = logging.getLogger(__name__)


@dataclass
class Compiled:
    graph: BaseGraph
    sequence: pl.Sequence


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
        self, device: Device, compiler: BaseGraphCompiler[GraphType], path: str | None
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
                        "sequence": instance.sequence.to_abstract_repr(),
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

        Once this method have succeeded, the results are stored in `self.sequences`.
        """
        if len(self.graphs) == 0:
            raise Exception("No graphs to compile, did you forget to call `import_graphs`?")
        if filter is None:
            filter = lambda _graph, sequence, _index: True  # noqa: E731
        self.sequences = []
        for graph in self.graphs:
            try:
                sequence = graph.compute_sequence()
            except ValueError as e:
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

    def add_graphs(self, graphs: Sequence[GraphType]) -> None:
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
    async def run(self) -> list[ProcessedData]:
        """
        Run compiled graphs.

        You will need to call `self.compile` first, to make sure that the graphs are compiled.

        Returns:
            A list of processed data from the graphs inserted with `add_graphs`. Note that the
            list may be shorter than the number of graphs added with `add_graphs`, as not all
            graphs may be compiled or executed on a given device. In fact, there are categories
            of graphs that may never be compiled by this library, regardless of device, due to
            geometric constraints.
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
        path: str,
        compiler: BaseGraphCompiler[GraphType],
        device: Device = pl.devices.AnalogDevice,
    ):
        super().__init__(path=path, device=device, compiler=compiler)
        self.graphs: list[BaseGraph]
        self.device = device

    async def run(self, max_qubits: int = 8) -> list[ProcessedData]:
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
            return []
        processed_data = []
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
            states = cast(dict[str, Any], simul.run().sample_final_state())
            logger.debug("Execution of compiled graph # %s complete", id)
            processed_data.append(
                ProcessedData(
                    sequence=compiled.sequence, state_dict=states, target=compiled.graph.target
                )
            )

        logger.debug("Emulation step complete, %s compiled graphs executed", len(processed_data))
        super().save(snapshot=processed_data)

        return processed_data


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
        path: str,
        compiler: BaseGraphCompiler[GraphType],
        device: Device = pl.devices.AnalogDevice,
    ):
        super().__init__(path=path, device=device, compiler=compiler)
        self.graphs: list[BaseGraph]
        self.device = device

    async def run(self, max_qubits: int = 10, dt: int = 10) -> list[ProcessedData]:
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
            return []
        backend = emu_mps.MPSBackend()
        processed_data = []
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
            states: dict[str, Any] = backend.run(compiled.sequence, config)[observable.name][
                cutoff_duration
            ]
            logger.debug("Execution of compiled graph # %s complete", id)
            processed_data.append(
                ProcessedData(
                    sequence=compiled.sequence, state_dict=states, target=compiled.graph.target
                )
            )

        logger.debug("Emulation step complete, %s compiled graphs executed", len(processed_data))
        super().save(snapshot=processed_data)

        return processed_data


class QPUExtractor(BaseExtractor[GraphType]):
    """
    A Extractor that uses a distant physical QPU to run sequences
    compiled from graphs.

    Performance note: as of this writing, the waiting lines for a QPU
    may be very long. You may use this Extractor to resume your workflow
    with a computation that has been previously started.

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
        path: str,
        compiler: BaseGraphCompiler[GraphType],
        project_id: str,
        username: str,
        password: str | None = None,
        device_name: str = "FRESNEL",
        batch_id: list[str] | None = None,
    ):
        sdk = SDK(username=username, project_id=project_id, password=password)

        # Fetch the latest list of QPUs
        specs = sdk.get_device_specs_dict()
        device = cast(Device, deserialize_device(specs[device_name]))

        super().__init__(path=path, device=device, compiler=compiler)
        self._sdk = sdk
        self._batch_id = batch_id

    @property
    def batch_ids(self) -> list[str] | None:
        return self._batch_id

    async def run(self) -> list[ProcessedData]:
        if len(self.sequences) == 0:
            logger.warning("No sequences to run, did you forget to call compile()?")
            return []

        batches: list[Batch] = []
        if self._batch_id is None:
            # Enqueue jobs.
            self._batch_ids = []
            for compiled in self.sequences:
                logger.debug("Executing compiled graph #%s", id)
                batch = self._sdk.create_batch(
                    compiled.sequence.to_abstract_repr(),
                    # Note: The SDK API doesn't support runs longer than 500 jobs.
                    # If we want to add more runs, we'll need to split them across
                    # several jobs.
                    jobs=[{"runs": 500}],
                    wait=False,
                )
                logger.info(
                    "Remote execution of compiled graph #%s starting, batched with id %s",
                    id,
                    batch.id,
                )
                batches.append(batch)
                self._batch_ids.append(batch.id)
            logger.info(
                "All %s jobs enqueued for remote execution, with ids %s",
                len(batches),
                self._batch_ids,
            )
        else:
            # Get jobs back from the cloud API.
            for batch_id in self._batch_id:
                batches.append(self._sdk.get_batch(batch_id))

        # Now wait until all batches are complete.
        pending_batches: dict[str, Batch] = {batch.id: batch for batch in batches}
        completed_batches: dict[str, Batch] = {}
        while len(pending_batches) > 0:
            # Fetch up to 100 pending batches (limit imposed by the SDK).
            check_ids = []
            for batch in pending_batches.values():
                check_ids.append(batch.id)
                if len(check_ids) >= 100:
                    break
            # Update their status.
            check_batches = self._sdk.get_batches(
                filters=BatchFilters(id=check_ids)
            )  # Ideally, this should be async, cf. https://github.com/pasqal-io/pasqal-cloud/issues/162.
            for batch in check_batches.results:
                assert isinstance(batch, Batch)
                if batch.status in {"PENDING", "RUNNING"}:
                    logger.debug("Job %s is now complete", batch.id)
                    pending_batches.pop(batch.id)
                    completed_batches[batch.id] = batch
            if len(pending_batches) > 0:
                logger.debug("%s jobs are still incomplete", len(pending_batches))
                await sleep(delay=2)

        logger.info("All jobs complete, %s sequences executed", len(batches))

        # At this point, all batches are complete.
        # Now collect data. We rely upon the fact
        # that we enqueued exactly one batch per sequence, in the same order.
        processed_data: list[ProcessedData] = []
        for i, original_batch in enumerate(batches):
            # Note: There's only one job per batch.
            assert len(original_batch.jobs) == 1
            batch = completed_batches[original_batch.id]
            assert len(original_batch.jobs) == 1

            for _, job in batch.jobs.items():
                if job.status == "DONE":
                    state_dict = job.result
                    assert state_dict is not None
                    processed_data.append(
                        ProcessedData(
                            sequence=self.sequences[i].sequence,
                            state_dict=state_dict,
                            target=self.sequences[i].graph.target,
                        )
                    )
                else:
                    # If some sequences failed, let's proceed as well as we can.
                    logger.warning(
                        "Job %s failed with errors %s, skipping", i, job.status, job.errors
                    )

        logger.info("All jobs complete, %s sequences succeeded", len(processed_data))
        super().save(snapshot=processed_data)
        return processed_data
