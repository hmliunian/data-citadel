"""All review CLI operations use the public HTTP API."""
import argparse
import json
import sys
import webbrowser
import httpx

from .api import CitadelClient


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data Citadel HTTP client")
    parser.add_argument("--url", default="http://127.0.0.1:8770")
    parser.add_argument("--run-id", default="grasp")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("runs")
    commands.add_parser("gui")
    for name in ("episodes", "report", "batch"):
        command = commands.add_parser(name)
        command.add_argument("--split", choices=["development", "holdout"], default="development")
        if name == "batch":
            command.add_argument("--limit", type=int)
            command.add_argument("--retry-failed", action="store_true")
    for name in ("review", "preview"):
        command = commands.add_parser(name)
        command.add_argument("episode_id")
        command.add_argument("--retry-failed", action="store_true")
        command.add_argument("--wait", action="store_true")
    for name in ("job", "output", "result"):
        commands.add_parser(name).add_argument("id")
    commands.add_parser("freeze")
    args = parser.parse_args(argv)
    if args.command == "gui":
        url = args.url.rstrip("/") + "/?run=" + args.run_id
        print(url)
        return 0 if webbrowser.open(url) else 1
    with CitadelClient(args.url) as client:
        if args.command == "runs":
            value = client.runs()
        elif args.command in ("episodes", "report"):
            value = getattr(client, args.command)(args.run_id, args.split)
        elif args.command in ("review", "preview"):
            value = client.submit(args.run_id, args.episode_id, preview=args.command == "preview",
                                  retry_failed=args.retry_failed)
            if args.wait:
                job = client.wait(value["job_id"])
                value = client.output(job["job_id"]) if job["status"] == "succeeded" or job["result_id"] else job
        elif args.command == "batch":
            value = client.batch(args.run_id, args.split, limit=args.limit, retry_failed=args.retry_failed)
        elif args.command == "result":
            value = client.result(args.run_id, args.id)
        elif args.command in ("job", "output"):
            value = getattr(client, args.command)(args.id)
        else:
            value = client.freeze(args.run_id)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return int(isinstance(value, dict) and value.get("status") == "failed")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (httpx.HTTPError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
