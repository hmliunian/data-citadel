"""Small file operations shared by storage adapters."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def write_frozen(path: Path, value):
    """Publish immutable content, accepting only an identical previous write."""
    try:
        write(path, value)
    except FileExistsError:
        if read(path) != value:
            raise ValueError("Frozen content changed: " + str(path))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()
