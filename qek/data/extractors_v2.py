"""
High-Level API to compile raw data (graphs) and process it on a quantum device, either a local emulator,
a remote emulator or a physical QPU.

Unlike `qek.data.extractors`, this module only speaks Pulser: any `pulser.backend.remote.RemoteConnection`
(e.g. `pasqal_cloud.PasqalCloudConnection`) and any `RemoteBackend` will do, so nothing here
depends on the pasqal-cloud SDK's own job API.
"""

import abc
import asyncio
import logging
import time
from typing import Any, Generator, Generic, Type
from pulser.backend import QPUBackend
from pulser.backend.remote import RemoteConnection, BatchStatus, RemoteBackend, RemoteResults
from pathlib import Path
import pulser as pl
from pulser.devices import Device

from qek.data.extractors import BaseExtracted, Compiled, SyncExtracted, GraphType, BaseExtractor
from qek.data.graphs import BaseGraph, BaseGraphCompiler
from qek.data.processed_data import ProcessedData

logger = logging.getLogger(__name__)

# How many seconds to sleep while waiting for the results from the cloud.
SLEEP_DELAY_S = 2

# Batch statuses that mean "come back later".
_PENDING_STATUSES = {BatchStatus.PENDING, BatchStatus.RUNNING}


class RemoteExtracted(BaseExtracted):
    """
    Data extracted from a remote connection, i.e. we need to wait for a remote server.

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
            batch_ids: The ids of the batches on the remote connection, in the same
                order as `compiled`, one batch per compiled graph.
            connection: The connection on which the batches were submitted.
            path: If provided, a path at which to save the results once they're available.
        """
        self._compiled = compiled
        self._batch_ids = batch_ids
        self._results: SyncExtracted | None = None
        self._path = path
        self._connection = connection

    def _poll(self) -> Generator[None, None, None]:
        """
        Poll the remote connection until all batches are complete, ingesting the results.

        Yields once per round, leaving it to the caller to wait between rounds (blocking
        or not). Yields nothing at all if the results are already available.
        """
        if self._results is not None:
            # Results are already available.
            return
        pending = {
            bid: RemoteResults(batch_id=bid, connection=self._connection) for bid in self._batch_ids
        }
        completed: dict[str, RemoteResults] = {}
        while len(pending) > 0:
            yield
            for bid, remote_results in list(pending.items()):
                if remote_results.get_batch_status() not in _PENDING_STATUSES:
                    logger.debug("Batch %s is now complete", bid)
                    completed[bid] = pending.pop(bid)

        # At this point, all batches are complete.
        self._ingest(completed)

    def _wait(self) -> None:
        """
        Wait synchronously until remote execution is ready.

        This WILL BLOCK your main thread, possibly for a very long time.
        """
        for _ in self._poll():
            time.sleep(SLEEP_DELAY_S)

    def __await__(self) -> Generator[Any, Any, None]:
        """
        Wait asynchronously until remote execution is ready.

        This will NOT block your main thread, so this method is strongly recommended
        for use on a server or an interactive application.

        Example:
            await extracted
        """
        for _ in self._poll():
            yield from asyncio.sleep(SLEEP_DELAY_S).__await__()

    def _ingest(self, completed: dict[str, RemoteResults]) -> None:
        """
        Ingest data received from the remote server.

        No I/O.
        """
        assert len(completed) == len(self._batch_ids)

        raw_data = []
        targets: list[int] = []
        sequences = []
        states = []
        for i, id in enumerate(self._batch_ids):
            compiled = self._compiled[i]
            # We submit exactly one job per compiled graph.
            results = list(completed[id].get_available_results().values())
            if len(results) != 1:
                # If some sequences failed, let's skip them and proceed as well as we can.
                logger.warning(
                    "Batch %s (graph %s) returned %s results instead of 1, skipping",
                    id,
                    compiled.graph.id,
                    len(results),
                )
                continue
            try:
                bitstrings = results[0].final_bitstrings
            except RuntimeError as e:
                logger.warning(
                    "Batch %s (graph %s) did not return a usable state (%s), skipping",
                    id,
                    compiled.graph.id,
                    e,
                )
                continue
            raw_data.append(compiled.graph)
            if compiled.graph.target is not None:
                targets.append(compiled.graph.target)
            sequences.append(compiled.sequence)
            states.append(bitstrings)
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


class BaseRemoteExtractorV2(BaseExtractor[GraphType], Generic[GraphType]):
    """
    An Extractor that runs sequences compiled from graphs on a remote Quantum Device,
    reachable through any Pulser `RemoteConnection`.

    Performance note (servers and interactive applications only):
        If your code is meant to be executed as part of an interactive application or
        a server, you should consider calling `await extracted` before your first call
        to any of the methods of `extracted`. Otherwise, you will block the main thread.

        If you are running this as part of an experiment, a Jupyter notebook, etc. you
        may ignore this performance note.

    Args:
        compiler: A graph compiler, in charge of converting graphs to Pulser Sequences.
        connection: An open connection to the remote API, e.g.
            `pasqal_cloud.PasqalCloudConnection`.
        device: The device to compile for. If unspecified, fetch `device_name` from
            `connection`.
        device_name: The name of the device to fetch from `connection`. As of this writing,
            the default value of "FRESNEL" represents the latest QPU available through
            the Pasqal Cloud API. Ignored if `device` is specified.
        batch_ids: Use this to resume a workflow e.g. after turning off
            your computer while the QPU was executing your sequences.
            Warning: A batch started with one extractor MUST NOT be resumed
            with a different extractor.
        path: Path to store the result of the run, for future uses.
            To reload the result of a previous run, use `LoadExtractor`.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        connection: RemoteConnection,
        device: Device | None = None,
        device_name: str = "FRESNEL",
        batch_ids: list[str] | None = None,
        path: Path | None = None,
    ):
        if device is None:
            # Fetch the latest specs of the device.
            device = connection.fetch_available_devices()[device_name]

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
        **backend_kwargs: Any,
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
            # Enqueue one batch per compiled graph.
            self._batch_ids = []
            for compiled in self.sequences:
                logger.debug("Enqueuing execution of compiled graph #%s", compiled.graph.id)
                remote_results = backend_class(
                    compiled.sequence, self._connection, **backend_kwargs
                ).run(job_params=[{"runs": max_runs}], wait=False)
                batch_id = remote_results.batch_id
                logger.info(
                    "Remote execution of compiled graph #%s starting, batch with id %s",
                    compiled.graph.id,
                    batch_id,
                )
                self._batch_ids.append(batch_id)
            logger.info(
                "All %s batches enqueued for remote execution, with ids %s",
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


class RemoteExtractorV2(BaseRemoteExtractorV2[GraphType]):
    """
    An Extractor that runs sequences compiled from graphs on a remote backend.

    By default, it runs on a QPU (`QPUBackend`). To run on a remote emulator instead, pass
    the corresponding backend class, e.g.:

        RemoteExtractorV2(compiler, connection, backend_class=pasqal_cloud.RemoteMPSBackend)

    Performance note:
        as of this writing, the waiting lines for a QPU
        may be very long. You may use this Extractor to resume your workflow
        with a computation that has been previously started, by passing `batch_ids`.

    Performance note (servers and interactive applications only):
        If your code is meant to be executed as part of an interactive application or
        a server, you should consider calling `await extracted` before your first call
        to any of the methods of `extracted`. Otherwise, you will block the main thread.

        If you are running this as part of an experiment, a Jupyter notebook, etc. you
        may ignore this performance note.

    Args:
        backend_class: The Pulser remote backend to execute the sequences on. It must be
            compatible with `connection`.
        backend_kwargs: Any additional arguments for `backend_class`, e.g. `config`.

        See `BaseRemoteExtractorV2` for the other arguments.
    """

    def __init__(
        self,
        compiler: BaseGraphCompiler[GraphType],
        connection: RemoteConnection,
        backend_class: Type[RemoteBackend] = QPUBackend,
        device: Device | None = None,
        device_name: str = "FRESNEL",
        batch_ids: list[str] | None = None,
        path: Path | None = None,
        **backend_kwargs: Any,
    ):
        super().__init__(
            compiler=compiler,
            connection=connection,
            device=device,
            device_name=device_name,
            batch_ids=batch_ids,
            path=path,
        )
        self._backend_class = backend_class
        self._backend_kwargs = backend_kwargs

    def run(self) -> RemoteExtracted:
        return self._run(backend_class=self._backend_class, **self._backend_kwargs)
