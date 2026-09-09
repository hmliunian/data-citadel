"""Small, explicit pilot commands; only run issues paid model requests."""
import argparse
import json
from pathlib import Path

from .data import DEFAULT_DATASET, load_manifest, prepare
from .media import prepare_media
from .runner import freeze, report, run_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preparation.add_argument("--task-code", default="DL-8GY1IC")
    preparation.add_argument("--experts", type=int, default=3)
    sample = commands.add_parser("sample")
    sample.add_argument("--split", choices=("experts", "development"), default="development")
    run = commands.add_parser("run")
    run.add_argument("--split", choices=("development", "holdout"), required=True)
    run.add_argument("--route", choices=("A", "B"), required=True)
    run.add_argument("--retry-failed", action="store_true")
    commands.add_parser("freeze")
    commands.add_parser("report")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.command == "prepare":
        manifest = prepare(args.dataset, args.run_dir, args.task_code, args.experts)
        print(json.dumps({"task_code": manifest["task_code"], "splits": manifest["splits"]}, indent=2))
    elif args.command == "sample":
        manifest = load_manifest(args.run_dir)
        for episode_id in manifest["splits"][args.split]:
            media = prepare_media(args.run_dir, manifest["episodes"][episode_id], manifest["sampling"])
            print(json.dumps({"episode_id": episode_id, "frames": len(media["frames"]),
                              "warnings": media["warnings"]}), flush=True)
    elif args.command == "run":
        for result in run_split(args.run_dir, args.split, args.route, retry_failed=args.retry_failed):
            print(json.dumps({k: result[k] for k in ("episode_id", "split", "route", "gt", "label", "status", "reason")},
                             ensure_ascii=False), flush=True)
    elif args.command == "freeze":
        print(json.dumps(freeze(args.run_dir), ensure_ascii=False, indent=2))
    elif args.command == "report":
        folder, summary = report(args.run_dir)
        print(json.dumps({"report_dir": str(folder), **{k: v for k, v in summary.items() if k != "calls"}},
                         ensure_ascii=False, indent=2))
    else:
        import uvicorn
        from .web import create_app
        uvicorn.run(create_app(args.run_dir), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
