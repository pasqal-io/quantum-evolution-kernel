"""
High-Level API to compile raw data (graphs) and process it on a quantum device, either a local emulator,
a remote emulator or a physical QPI.
"""

import abc
import asyncio
import logging
import time
from typing import Any, Generator, Generic, cast, Type
from pulser.backend import QPUBackend, Results
from pulser.backend.remote import RemoteConnection, BatchStatus, RemoteBackend, RemoteResults
from pulser_pasqal.backends import EmuMPSBackend as RemoteMPSBackend
from pathlib import Path
import pulser as pl
from pulser.devices import Device
from pulser.json.abstract_repr.deserializer import deserialize_device

from qek.data.extractors import BaseExtracted, Compiled, SyncExtracted, GraphType, BaseExtractor
from qek.data.graphs import BaseGraph, BaseGraphCompiler
from qek.data.processed_data import ProcessedData

logger = logging.getLogger(__name__)

# How many seconds to sleep while waiting for the results from the cloud.
SLEEP_DELAY_S = 2


class RemoteExtracted(BaseExtracted):
    """
    Data extracted from remote API, i.e. we need wait for a remote server.

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
        connection: RemoteConnection,
        path: Path | None = None,
    ):
        """
        Prepare for reception of data.

        Arguments:
            compiled: The result of compiling a set of graphs.
            job_ids: The ids of the jobs on the cloud API, in the same order as `compiled`.
            path: If provided, a path at which to save the results once they're available.
        """
        self._compiled = compiled
        self._batch_ids = batch_ids
        self._results: SyncExtracted | None = None
        self._path = path
        self._connection = connection

    def _wait(self) -> None:
        """
        Wait synchronously until remote execution is ready.

        This WILL BLOCK your main thread, possibly for a very long time.
        """
        if self._results is not None:
            # Results are already available.
            return
        pending_batch_ids: set[str] = set(self._batch_ids)
        all_remote_results = {bid: RemoteResults(batch_id=bid,connection=self._connection) for bid in pending_batch_ids} 
        completed_batchs: dict[str, Results] = {}
        while len(pending_batch_ids) > 0:
            time.sleep(SLEEP_DELAY_S)
            # Update their status.
            for bid in pending_batch_ids:
                remote_results = all_remote_results[bid]
                batch_status = remote_results.get_batch_status()
                if batch_status not in {BatchStatus.PENDING, BatchStatus.RUNNING}:
                    logger.debug("Batch %s is now complete", bid)
                    pending_batch_ids.discard(bid)
                    completed_batchs[bid] = remote_results

        # At this point, all jobs are complete.
        self._ingest(completed_batchs)

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
        all_remote_results = {bid: RemoteResults(batch_id=bid,connection=self._connection) for bid in pending_batch_ids} 
        completed_batchs: dict[str, Results] = {}
        while len(pending_batch_ids) > 0:
            yield from asyncio.sleep(SLEEP_DELAY_S).__await__()
            # Update their status.
            for bid in pending_batch_ids:
                remote_results = all_remote_results[bid]
                batch_status = remote_results.get_batch_status()
                if batch_status not in {BatchStatus.PENDING, BatchStatus.RUNNING}:
                    logger.debug("Batch %s is now complete", bid)
                    pending_batch_ids.discard(bid)
                    completed_batchs[bid] = remote_results

        # At this point, all jobs are complete.
        self._ingest(completed_batchs)

    def _ingest(self, completed_batch: dict[str, RemoteResults]) -> None:
        """
        Ingest data received from the remote server.

        No I/O.
        """
        assert len(completed_batch) == len(self._batch_ids)

        raw_data = []
        targets: list[int] = []
        sequences = []
        all_bitstrings = []
        for i, id in enumerate(self._batch_ids):
            batch_results = completed_batch[id]
            compiled = self._compiled[i]
            results = list(batch_results.get_available_results().values())
            if len(results) == 1:
                job_results = results[0]
                bitstrings = self._state_extractor(job_results.final_bitstrings, compiled.sequence)
                if bitstrings is None:
                    logger.warning(
                        "Job %s (graph %s) did not return a usable state, skipping",
                        i,
                        compiled.graph.id,
                    )
                    continue
                raw_data.append(compiled.graph)
                if compiled.graph.target is not None:
                    targets.append(compiled.graph.target)
                sequences.append(compiled.sequence)
                all_bitstrings.append(bitstrings)
            else:
                # If some sequences failed, let's skip them and proceed as well as we can.
                logger.warning(
                    "Job %s (graph %s) failed, skipping",
                    i,
                    compiled.graph.id
                )
        self._results = SyncExtracted(
            raw_data=raw_data, targets=targets, sequences=sequences, states=all_bitstrings
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


class BaseRemoteExtractorV2(BaseExtractor[GraphType], Generic[GraphType]):
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
        batch_ids: Use this to resume a workflow e.g. after turning off
            your computer while the QPU was executing your sequences.
            Warning: A job started with one executor MUST NOT be resumed
            with a different executor.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        connection: RemoteConnection,
        batch_ids: list[str] | None = None,
        device_name: str = "FRESNEL",
        path: Path | None = None,
    ):

        # Fetch the latest list of QPUs
        specs = connection.fetch_available_devices()
        device = cast(Device, deserialize_device(specs[device_name]))

        super().__init__(device=device, compiler=compiler, path=path)
        self._connection = connection
        self._batch_ids: list[str] | None = batch_ids

    @property
    def batch_ids(self) -> list[str] | None:
        return self._batch_ids

    @abc.abstractmethod
    def run(
        self,
    ) -> RemoteExtracted:
        """
        Launch the extraction.
        """
        raise NotImplementedError()

    def _run(
        self,
        backend_class: Type[RemoteBackend],
    ) -> RemoteExtracted:
        if len(self.sequences) == 0:
            logger.warning("No sequences to run, did you forget to call compile()?")
            return RemoteExtracted(
                compiled=[],
                batch_ids=[],
                connection=self._connection,
                path=self.path,
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
                remote_results = backend_class(compiled.sequence, self._connection).run(
                    jobs_params=[{"runs": max_runs}],
                    wait=False,
                )
                batch_id = remote_results.batch_id
                logger.info(
                    "Remote execution of compiled graph #%s starting, job with id %s",
                    compiled.graph.id,
                    batch_id,
                )
                self._batch_ids.append(batch_id)
            logger.info(
                "All %s jobs enqueued for remote execution, with ids %s",
                len(self._batch_ids),
                self._batch_ids,
            )
        assert len(self._batch_ids) == len(self.sequences)

        return RemoteExtracted(
            compiled=self.sequences,
            batch_ids=self._batch_ids,
            connection=self._connection,
            path=self.path,
        )


class RemoteQPUExtractorV2(BaseRemoteExtractorV2[GraphType]):
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
        job_id: Use this to resume a workflow e.g. after turning off
            your computer while the QPU was executing your sequences.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        connection: RemoteConnection,
        batch_ids: list[str] | None = None,
        device_name: str = "FRESNEL",
        path: Path | None = None,
    ):
        super().__init__(
            compiler=compiler,
            connection=connection,
            batch_ids=batch_ids,
            device_name=device_name,
            path=path,
        )

    def run(self) -> RemoteExtracted:
        return self._run(backend_class=QPUBackend)


class RemoteEmuMPSExtractorV2(BaseRemoteExtractorV2[GraphType]):
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
        job_id: Use this to resume a workflow e.g. after turning off
            your computer while the QPU was executing your sequences.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        connection: RemoteConnection,
        batch_ids: list[str] | None = None,
        device_name: str = "FRESNEL",
        path: Path | None = None,
    ):
        super().__init__(
            compiler=compiler,
            connection=connection,
            batch_ids=batch_ids,
            device_name=device_name,
            path=path,
        )

    def run(self) -> RemoteExtracted:
        return self._run(
            backend_class=RemoteMPSBackend,
        )
