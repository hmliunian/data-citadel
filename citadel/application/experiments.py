"""Experiment scope and GT evaluation, separate from model inputs."""
from citadel.domain.errors import GateError


def counts(items):
    complete = [r for r in items if r["status"] == "completed" and r["gt"]]
    matched = sum(r["label"] == r["gt"] for r in complete)
    labeled = sum(bool(r["gt"]) for r in items)
    return {
        "total": len(items), "labeled": labeled,
        **{s: sum(r["status"] == s for r in items)
           for s in ("completed", "needs_review", "failed", "not_run")},
        "matched": matched,
        "false_accept": sum(r["gt"] == "incorrect" and r["label"] == "correct" for r in complete),
        "false_reject": sum(r["gt"] == "correct" and r["label"] == "incorrect" for r in complete),
        "coverage": len(complete) / labeled if labeled else None,
        "accuracy_on_completed": matched / len(complete) if complete else None,
    }


class ExperimentService:
    def __init__(self, source, artifacts, resources):
        self.source, self.artifacts, self.resources = source, artifacts, resources

    def split_of(self, episode_id):
        if episode_id not in self.source.records():
            raise KeyError("Unknown episode ID")
        for split, ids in self.source.manifest["splits"].items():
            if episode_id in ids:
                return split
        raise ValueError("Duplicate episode is excluded from this run")

    def gate(self, split, snapshot):
        if split not in self.source.manifest["splits"]:
            raise ValueError("Split must be development or holdout")
        if split == "holdout":
            frozen = self.artifacts.frozen()
            if not frozen:
                raise GateError("Freeze after development before opening holdout")
            if frozen["configuration_sha256"] != snapshot.sha256:
                raise GateError("Frozen configuration changed; use a new run")
            for code, expected in frozen["resources"].items():
                if self.resources.get(code)["sha256"] != expected:
                    raise GateError("Frozen task references changed")

    def episodes(self, split, snapshot):
        self.gate(split, snapshot)
        rows = self.source.records()
        output = []
        for episode_id in self.source.manifest["splits"][split]:
            row = rows[episode_id]
            result = self.artifacts.latest(episode_id, snapshot.sha256)
            output.append({"episode_id": episode_id, "task_code": row["task_code"],
                           "gt": row["gt"], "gt_reason": row["gt_reason"], "split": split,
                           "status": result["status"] if result else "not_run",
                           "label": result.get("label") if result else None,
                           "result_id": result["result_id"] if result else None})
        return output

    def decorate(self, result):
        row = self.source.records()[result["episode_id"]]
        return {**result, "evaluation": {"gt": row["gt"], "gt_reason": row["gt_reason"]}}

    def freeze(self, snapshot):
        if self.artifacts.frozen():
            self.gate("holdout", snapshot)
            return self.artifacts.frozen()
        if any(row["status"] in ("failed", "not_run")
               for row in self.episodes("development", snapshot)):
            raise GateError("Review all development episodes and resolve execution errors first")
        codes = {row["task_code"] for row in self.source.records().values()}
        return self.artifacts.freeze({"configuration_sha256": snapshot.sha256,
                                      "configuration": snapshot.data,
                                      "resources": {code: self.resources.get(code)["sha256"]
                                                    for code in sorted(codes)}})

    def report(self, split, snapshot):
        rows = self.episodes(split, snapshot)
        reasons = {r["gt_reason"] or r["gt"] or "unlabeled" for r in rows}
        return {"split": split, "configuration_sha256": snapshot.sha256,
                "counts": counts(rows), "episodes": rows,
                "by_gt_reason": {reason: counts([r for r in rows if
                                 (r["gt_reason"] or r["gt"] or "unlabeled") == reason])
                                 for reason in sorted(reasons)},
                "by_task": {code: counts([r for r in rows if r["task_code"] == code])
                            for code in sorted({r["task_code"] for r in rows})},
                "usage": self.artifacts.usage(snapshot.sha256, split)}
