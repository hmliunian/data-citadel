"""Persistent task state and atomic deduplication for the local scheduler."""
from contextlib import contextmanager
import json
import sqlite3
import time
import uuid

from citadel.domain.errors import BusyError
from .files import fingerprint


class JobRepository:
    def __init__(self, path, max_pending=128):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.max_pending = path, max_pending
        with self.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, identity TEXT NOT NULL, run_id TEXT NOT NULL,
                episode_id TEXT NOT NULL, kind TEXT NOT NULL, snapshot TEXT NOT NULL,
                retry_failed INTEGER NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL,
                created_ns INTEGER NOT NULL, updated_ns INTEGER NOT NULL,
                result TEXT, error TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_identity ON jobs(identity, created_ns)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError("Unknown job ID")
        value = dict(row)
        for key in ("snapshot", "result", "error"):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        value["job_id"] = value.pop("id")
        value.pop("identity")
        return value

    def get(self, job_id):
        with self.connection() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def create(self, run_id, episode_id, kind, snapshot, retry_failed):
        return self.create_many(run_id, [episode_id], kind, snapshot, retry_failed)[0]

    def create_many(self, run_id, episode_ids, kind, snapshot, retry_failed):
        output = []
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            pending = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            for episode_id in episode_ids:
                identity = fingerprint([run_id, episode_id, kind, snapshot.sha256])
                previous = db.execute("SELECT * FROM jobs WHERE identity=? ORDER BY created_ns DESC LIMIT 1",
                                      (identity,)).fetchone()
                if previous and not (retry_failed and previous["status"] == "failed"):
                    output.append((self.decode(previous), False))
                    continue
                if pending >= self.max_pending:
                    raise BusyError("The task queue is full; this submission was not queued")
                job_id, stamp = uuid.uuid4().hex, time.time_ns()
                db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (job_id, identity, run_id, episode_id, kind, snapshot.payload,
                            retry_failed, "queued", "queued", stamp, stamp, None, None))
                output.append((self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()), True))
                pending += 1
        return output

    def update(self, job_id, *, status=None, stage=None, result=None, error=None):
        values = {k: v for k, v in {"status": status, "stage": stage,
                                    "result": result, "error": error}.items() if v is not None}
        for key in ("result", "error"):
            if key in values:
                values[key] = json.dumps(values[key], ensure_ascii=False)
        values["updated_ns"] = time.time_ns()
        with self.connection() as db:
            db.execute("UPDATE jobs SET " + ",".join(key + "=?" for key in values) + " WHERE id=?",
                       (*values.values(), job_id))

    def active(self, run_id):
        with self.connection() as db:
            return db.execute("SELECT COUNT(*) FROM jobs WHERE run_id=? AND status IN ('queued','running')",
                              (run_id,)).fetchone()[0]

    def recover(self):
        with self.connection() as db:
            db.execute("UPDATE jobs SET status='failed', error=?, updated_ns=? WHERE status='running'",
                       (json.dumps({"type": "Interrupted", "message": "Server stopped during execution; inspect call receipts before retrying"}),
                        time.time_ns()))
            return [row["id"] for row in db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_ns")]
