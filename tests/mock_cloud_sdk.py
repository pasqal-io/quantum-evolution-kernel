import json

from pasqal_cloud.job import CreateJob
from pasqal_cloud.device import BaseConfig, EmulatorType
from pasqal_cloud.batch import Batch
from pasqal_cloud.utils.responses import PaginatedResponse

from uuid import uuid4

from unittest.mock import MagicMock
from typing import Any


class MockSDK:
    """Helper class to mock the cloud SDK and skip the API calls.

    Warning: This implements only a small subset of the functions available
        in the cloud SDK; focusing on the functions used by qek.
        This should be extended to support all methods and moved to the
        pasqal-cloud repository so that it can be reused for all future
        libraries using the SDK.

    TODO:
      - mock execution of jobs and batches (pending => running => done)
      - set proper results for jobs (right now its entirely mocked)
    """

    def __init__(self) -> None:
        self.batches: dict[str, Batch] = {}

    def get_device_specs_dict(self) -> Any:
        with open("tests/fixtures/device_specs.json", "r") as f:
            return json.load(f)

    def create_batch(
        self,
        serialized_sequence: str,
        jobs: list[CreateJob],
        open: bool | None = None,
        emulator: EmulatorType | None = None,
        configuration: BaseConfig | None = None,
        wait: bool = False,
    ) -> Batch:
        batch_id = str(uuid4())
        batch = Batch(
            id=batch_id,
            open=bool(open),
            complete=bool(open),
            created_at="",
            updated_at="",
            device_type=emulator if emulator else "FRESNEL",
            project_id="",
            user_id="",
            status="DONE",
            jobs=[
                {
                    **j,
                    "batch_id": batch_id,
                    "id": str(uuid4()),
                    "project_id": "",
                    "status": "DONE",
                    "created_at": "",
                    "updated_at": "",
                }
                for j in jobs
            ],
            configuration=configuration,
            _client=MagicMock(),
        )

        self.batches[batch.id] = batch
        return batch

    def get_batches(self, *args: Any, **kwargs: Any) -> PaginatedResponse:
        return PaginatedResponse(
            results=list(self.batches.values()),
            total=len(self.batches.values()),
            offset=0,
        )
