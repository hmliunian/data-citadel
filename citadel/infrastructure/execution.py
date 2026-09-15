"""Bounded local execution with a single scheduler per server work directory."""
from concurrent.futures import ThreadPoolExecutor
import fcntl

from citadel.domain.errors import BusyError


class LocalExecutor:
    def __init__(self, root, workers):
        root.mkdir(parents=True, exist_ok=True)
        self._lock = (root / ".scheduler.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise BusyError("This work directory already has a scheduler") from exc
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="citadel")
        self.closed = False

    def submit(self, function, *args):
        return self.pool.submit(function, *args)

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True)
            self._lock.close()
            self.closed = True
