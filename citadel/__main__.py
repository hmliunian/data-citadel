"""Server and local dataset preparation entry points."""
import argparse
import json
from pathlib import Path

from .configuration import CONFIG_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data Citadel server")
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .infrastructure.datasets import prepare
        manifest = prepare(args.dataset, args.work_dir)
        print(json.dumps({key: len(ids) for key, ids in manifest["splits"].items()}))
        return 0
    import uvicorn
    from .bootstrap import build_runtime
    from .server.app import create_app
    runtime = build_runtime(args.config)
    try:
        uvicorn.run(create_app(runtime), host=args.host or runtime.settings.host,
                    port=args.port or runtime.settings.port)
    finally:
        runtime.jobs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
