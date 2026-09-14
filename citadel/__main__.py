"""Run with python -m citadel; output stays in the selected work directory."""
import argparse
import json
import sys
import uuid
from pathlib import Path

from .infrastructure.datasets import prepare
from .infrastructure.files import write
from .service import Service


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data Citadel atomic-task review")
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/grasp_v1"))
    parser.add_argument("--profiles", type=Path, default=Path("config/tasks.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="Create a fixed GT split without model calls")
    setup.add_argument("--dataset", type=Path, required=True)
    setup.add_argument("--holdout-fraction", type=float, default=0.3)
    review = commands.add_parser("review", help="Review one candidate")
    review.add_argument("episode_id")
    review.add_argument("--retry-failed", action="store_true")
    run = commands.add_parser("run", help="Review a split with bounded concurrency; uncached candidates call Qwen")
    run.add_argument("--split", choices=["development", "holdout"], default="development")
    run.add_argument("--limit", type=int)
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--retry-failed", action="store_true")
    commands.add_parser("freeze", help="Freeze after all development candidates are processed")
    report = commands.add_parser("report", help="Export counts, errors, usage and candidate rows")
    report.add_argument("--split", choices=["development", "holdout"], default="development")
    serve = commands.add_parser("serve", help="Start the test window and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    def emit(value):
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
    if args.command == "prepare":
        manifest = prepare(args.dataset, args.work_dir, args.holdout_fraction)
        emit({"work_dir": str(args.work_dir), "split_counts":
              {k: len(v) for k, v in manifest["splits"].items()}})
        return 0
    service = Service(args.work_dir, args.profiles)
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(service), host=args.host, port=args.port)
    elif args.command == "review":
        result = service.review(args.episode_id, args.retry_failed)
        emit(result)
        return int(result["status"] == "failed")
    elif args.command == "run":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        failed = False
        for result in service.run(args.split, workers=args.workers, limit=args.limit,
                                  retry_failed=args.retry_failed):
            failed |= result["status"] == "failed"
            emit({k: result.get(k) for k in ("episode_id", "status", "label", "reason", "cached")})
        return int(failed)
    elif args.command == "freeze":
        emit(service.freeze())
    elif args.command == "report":
        result = service.report(args.split)
        folder = service.work / "reports" / uuid.uuid4().hex
        write(folder / "report.json", result)
        with (folder / "results.jsonl").open("x") as stream:
            for row in result["episodes"]:
                try:
                    value = service.get_result(row["episode_id"])
                except KeyError:
                    value = row
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        emit({"path": str(folder), "counts": result["counts"], "usage": result["usage"]})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
