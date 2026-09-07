"""Command-line entry points sharing the same review service as the web API."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .evaluation import evaluate_manifest
from .experts import prepare_experiment
from .models import CitadelError
from .repository import EpisodeRepository
from .runtime import build_sampler, build_service, save_review, write_json
from .settings import Settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="data-citadel")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--experts", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory", help="统计本地样本及专家集缺额")
    prepare = commands.add_parser("prepare", help="按原子动作生成专家与独立测试清单")
    prepare.add_argument("--output-dir", type=Path, default=Path("config"))
    sample = commands.add_parser("sample", help="导出带时间戳的 JPEG 抽帧")
    sample.add_argument("episode_id")
    sample.add_argument("--strategy", choices=["uniform", "keyframes"], default="uniform")
    sample.add_argument("--interval", type=float, default=2.0)
    sample.add_argument("--output-dir", type=Path, required=True)
    review = commands.add_parser("review", help="审核一次采集并保存结果")
    review.add_argument("episode_id")
    review.add_argument("--strategy", choices=["uniform", "keyframes"], default="uniform")
    evaluate = commands.add_parser("evaluate", help="按人工标签清单评测")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--strategy", choices=["uniform", "keyframes"], default="uniform")
    evaluate.add_argument("--limit", type=int)
    serve = commands.add_parser("serve", help="启动本地 Web 页面和 API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    settings = Settings()
    overrides = {name: getattr(args, name) for name in ["dataset_root", "artifacts_dir"]
                 if getattr(args, name) is not None}
    if args.experts:
        overrides["experts_path"] = args.experts
    settings = replace(settings, **overrides)
    service = None
    try:
        if args.command in ("serve", "review", "evaluate"):
            service = build_service(settings)
        repository = service.repository if service else EpisodeRepository(settings.dataset_root)
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(settings, service), host=args.host, port=args.port)
            return 0
        if args.command == "inventory":
            output = repository.inventory()
        elif args.command == "prepare":
            output = prepare_experiment(repository)
            destinations = [args.output_dir / f"{name}.json" for name in output]
            if any(path.exists() for path in destinations):
                raise FileExistsError("输出清单已存在，请使用新的 --output-dir")
            for name, value in output.items():
                write_json(args.output_dir / f"{name}.json", value)
            output = {"output_dir": str(args.output_dir), "report": output["report"]}
        elif args.command == "sample":
            episode = repository.get(args.episode_id)
            video = build_sampler(settings).sample(
                episode, strategy=args.strategy, interval_s=args.interval
            )
            args.output_dir.mkdir(parents=True, exist_ok=False)
            frames = []
            for index, frame in enumerate(video.frames):
                filename = f"{index:04d}.jpg"
                (args.output_dir / filename).write_bytes(frame.jpeg)
                frames.append({"timestamp_s": frame.timestamp_s, "file": filename})
            output = {"episode_id": episode.episode_id, "duration_s": video.duration_s,
                      "strategy": video.strategy, "camera_topic": video.camera_topic,
                      "warnings": video.warnings, "motion": video.motion, "frames": frames}
            write_json(args.output_dir / "frames.json", output)
        elif args.command == "review":
            result = service.review(args.episode_id, strategy=args.strategy)
            path = save_review(result, settings.artifacts_dir)
            output = {"result": result.model_dump(), "artifact": str(path)}
        else:
            output = evaluate_manifest(
                service, service.repository, args.manifest, args.output_dir,
                strategy=args.strategy, limit=args.limit,
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (CitadelError, OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
