"""
Backoff-and-retry utilities.
"""

import logging
from time import sleep
from typing import Any, Final, Type

from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)


class PygRetrier:
    """
    Our test harness attempts to run tests concurrently, but the pyg dataset loader does not
    work well with concurrency.

    We work around this by simply retrying the loads a few times, until it succeeds.
    """

    def __init__(self, max_attempts: int = 3, name: str = "PygRetrier"):
        """
        Create a PygRetrier

        Arguments:
            max_attempts (optional): The max number of attempts to undertake before
                giving up. Defaults to 3.
            name (optional): A name to use during logging.
        """
        self._max_attempts: Final[int] = max_attempts
        self.name: Final[str] = name

    def insist(self, callback: Type[Dataset], **kwargs: Any) -> Dataset:
        """
        Attempt to call a function or constructor repeatedly until, hopefully,
        it works.
        """
        exn: FileNotFoundError | RuntimeError | OSError | None = None
        result = None
        for i in range(self._max_attempts):
            sleep(i * i)
            try:
                logger.debug("%s: attempt %s", self.name, i + 1)
                result = callback(**kwargs)  # type: ignore
                logger.debug("%s: attempt %s succeeded", self.name, i + 1)
                exn = None
                break
            except (FileNotFoundError, RuntimeError, OSError) as e:
                logger.warning("%s: attempt %s failed: %s", self.name, i + 1, e)
                exn = e
        if exn is not None:
            logger.warning("%s: all attempts failed, bailing out", self.name)
            raise exn
        assert result is not None
        return result
