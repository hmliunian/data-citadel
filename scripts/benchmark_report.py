"""Recompute benchmark scores and costs from saved provider receipts."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

from citadel.application.experiments import counts
from citadel.infrastructure.files import fingerprint, read

RESULT_GLOB = "[0-9a-f]" * 32 + ".json"


def price_usage(usage, tiers):
    if not usage or "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return {"list_cny": None, "cache_adjusted_cny": None, "cached_tokens": None}
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if min(prompt, completion, cached) < 0 or cached > prompt:
        raise ValueError("Invalid usage receipt")
    tier = next((row for row in tiers if prompt <= row[0]), None)
    if tier is None:
        raise ValueError("Usage exceeds published price tiers")
    _, input_rate, output_rate, cached_rate = tier
    ordinary = (prompt * input_rate + completion * output_rate) / 1_000_000
    adjusted = (None if cached and cached_rate is None else
                ((prompt - cached) * input_rate + cached * (cached_rate or 0) +
                 completion * output_rate) / 1_000_000)
    return {"list_cny": ordinary, "cache_adjusted_cny": adjusted, "cached_tokens": cached}


def percentile(values, fraction):
    if not values:
        return None
    data = sorted(values)
    point = (len(data) - 1) * fraction
    low, high = math.floor(point), math.ceil(point)
    return data[low] + (data[high] - data[low]) * (point - low)


def wilson(matched, total):
    z = 1.959963984540054
    proportion = matched / total
    center = (proportion + z * z / (2 * total)) / (1 + z * z / total)
    span = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / (1 + z * z / total)
    return [center - span, center + span]


def call_rows(work, model, price):
    rows = []
    for path in sorted((work / "models" / model / "calls").glob("*/request.json")):
        request = read(path)
        response_path, error_path = path.with_name("response.json"), path.with_name("error.json")
        response = read(response_path) if response_path.exists() else {}
        error = read(error_path) if error_path.exists() else {}
        body = response.get("body", {})
        usage, context = body.get("usage", {}), request.get("context", {})
        choice = (body.get("choices") or [{}])[0]
        rows.append({
            "model": model, "episode_id": context.get("episode_id"), "result_id": context.get("result_id"),
            "stage": context.get("stage", "task"), "attempt": request["attempt"], "call_id": path.parent.name,
            "started_at": request.get("created_at"), "http_status": response.get("status"),
            "error_type": error.get("type"), "returned_model": body.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "reasoning_content_present": bool((choice.get("message") or {}).get("reasoning_content")),
            "request_sha256": request["request_sha256"], "input_images": request["input_images"],
            "response_format": request["parameters"].get("response_format", {}).get("type", "text"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "elapsed_s": response.get("elapsed_s", error.get("elapsed_s")),
            **price_usage(usage, price["tiers"])})
    return sorted(rows, key=lambda row: (row["started_at"] or "", row["call_id"]))


def summarize(work):
    frozen = read(work / "experiment.json")
    manifest = frozen["manifest"]
    split = {episode_id: name for name, ids in manifest["splits"].items() for episode_id in ids}
    summaries, episodes, calls = [], [], []
    for model in frozen["plan"]["models"]:
        receipts = call_rows(work, model, frozen["pricing"]["models"][model])
        calls.extend(receipts)
        results = {path.stem: read(path) for path in (work / "models" / model / "results").glob(RESULT_GLOB)}
        rows, latencies = [], []
        for episode_id, source in manifest["episodes"].items():
            result = results.get(episode_id, {})
            attempts = [row for row in receipts if row["episode_id"] == episode_id]
            api_time = sum(row["elapsed_s"] or 0 for row in attempts)
            row = {"model": model, "episode_id": episode_id, "split": split.get(episode_id, "excluded"),
                   "gt": source["gt"], "gt_reason": source["gt_reason"], "task_code": source["task_code"],
                   "status": result.get("status", "not_run"), "label": result.get("label"),
                   "error_stage": result.get("error", {}).get("stage"),
                   "error_type": result.get("error", {}).get("type"), "result_id": result.get("result_id"),
                   "calls": len(attempts), "list_cny": sum(r["list_cny"] or 0 for r in attempts),
                   "unknown_cost_calls": sum(r["list_cny"] is None for r in attempts),
                   "api_elapsed_s": api_time, "wall_elapsed_s": result.get("elapsed_s"),
                   "pacing_s": result.get("pacing_s"), "issues": result.get("issues", [])}
            if result and result["status"] != "failed":
                latencies.append(api_time)
            rows.append(row)
        score = counts(rows)
        positives = [row for row in rows if row["gt"] == "correct"]
        negatives = [row for row in rows if row["gt"] == "incorrect"]
        total_cost = sum(row["list_cny"] or 0 for row in receipts)
        reasons = sorted({row["gt_reason"] or row["gt"] for row in rows})
        positive_pass = sum(row["label"] == "correct" and row["status"] == "completed" for row in positives)
        negative_reject = sum(row["label"] == "incorrect" and row["status"] == "completed" for row in negatives)
        summaries.append({
            "model": model, **score, "match_rate": score["matched"] / len(rows),
            "match_rate_wilson95": wilson(score["matched"], len(rows)),
            "positive_pass": positive_pass,
            "positive_total": len(positives),
            "negative_reject": negative_reject,
            "balanced_match_rate": (positive_pass / len(positives) + negative_reject / len(negatives)) / 2
                                   if positives and negatives else None,
            "negative_total": len(negatives), "calls": len(receipts),
            "retries": sum(row["attempt"] > 1 for row in receipts),
            **{key: sum(row[key] or 0 for row in receipts)
               for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")},
            "unknown_usage_calls": sum(row["list_cny"] is None for row in receipts),
            "unknown_cache_price_calls": sum(row["list_cny"] is not None and row["cache_adjusted_cny"] is None for row in receipts),
            "list_cny": total_cost,
            "cache_adjusted_known_cny": sum(row["cache_adjusted_cny"] or 0 for row in receipts),
            "mean_cny_per_episode": total_cost / max(1, len(results)),
            "mean_cny_per_matched": total_cost / score["matched"] if score["matched"] else None,
            "valid_latency_n": len(latencies), "api_p50_s": statistics.median(latencies) if latencies else None,
            "api_p95_s": percentile(latencies, .95),
            "returned_models": sorted({row["returned_model"] for row in receipts if row["returned_model"]}),
            "reasoning_response_calls": sum(row["reasoning_content_present"] for row in receipts),
            "response_formats": sorted({row["response_format"] for row in receipts}),
            "failures_by_stage": {stage: sum(row["error_stage"] == stage for row in rows)
                                  for stage in sorted({r["error_stage"] for r in rows if r["error_stage"]})},
            "by_reason": {reason: counts([row for row in rows if (row["gt_reason"] or row["gt"]) == reason]) for reason in reasons},
            "by_split": {name: counts([row for row in rows if row["split"] == name]) for name in manifest["splits"]},
            "by_stage": {stage: {"calls": len([row for row in receipts if row["stage"] == stage]),
                                 "list_cny": sum(row["list_cny"] or 0 for row in receipts if row["stage"] == stage),
                                 "total_tokens": sum(row["total_tokens"] or 0 for row in receipts if row["stage"] == stage)}
                         for stage in ("task", "quality")}})
        episodes.extend(rows)
    return {"experiment_sha256": frozen["sha256"],
            "state": "running" if any(row["not_run"] for row in summaries) else "finished",
            "models": summaries}, episodes, calls, frozen


def csv_text(rows):
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    return output.getvalue()


def audit_receipts(work, frozen):
    """Check actual redacted messages, provenance and terminal receipts before publishing."""
    if fingerprint({key: value for key, value in frozen.items() if key != "sha256"}) != frozen["sha256"]:
        raise ValueError("Experiment digest mismatch")
    expected = {episode_id: read(work / "inputs" / (episode_id + ".json"))
                for episode_id in frozen["input_audit"]}
    checked = 0
    for model in frozen["plan"]["models"]:
        results = {path.stem: read(path) for path in (work / "models" / model / "results").glob(RESULT_GLOB)}
        if set(results) != set(expected):
            raise ValueError("Cannot publish an incomplete model scope")
        for episode_id, result in results.items():
            if (result["configuration_sha256"] != frozen["sha256"]
                    or result["input_sha256"] != fingerprint(frozen["input_audit"][episode_id])):
                raise ValueError("Result provenance mismatch")
        for path in (work / "models" / model / "calls").glob("*/request.json"):
            request = read(path)
            context = request["context"]
            episode_id, stage = context["episode_id"], context["stage"]
            if (request["messages"] != expected[episode_id][stage]
                    or context["configuration_sha256"] != frozen["sha256"]
                    or context["result_id"] != results[episode_id]["result_id"]
                    or request["model"] != model):
                raise ValueError("Request differs from frozen input or result")
            if not (path.with_name("response.json").exists() or path.with_name("error.json").exists()):
                raise ValueError("Request has no terminal receipt")
            checked += 1
    return {"requests_checked": checked, "identical_messages_per_episode_and_stage": True,
            "result_scope_and_provenance_verified": True}


def table(headers, rows):
    def line(values):
        return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *[line(row) for row in rows]])


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(summary, frozen):
    models = sorted(summary["models"], key=lambda row: (-row["matched"], row["false_accept"], row["list_cny"]))
    best = models[0]
    frames = [row["frames"] for row in frozen["input_audit"].values()]
    sections = [
        "# 7 个 Qwen 模型的费用与审核效果 benchmark",
        f"本轮 {len(models)} 个模型各完成 {best['total']} 条固定数据的测试。按完整数据分母计算，"
        f"`{best['model']}` 命中最多，为 **{best['matched']}/{best['total']}"
        f"（{best['match_rate']:.1%}）**。这是当前 prompt、采样和双调用流程的回归结果。",
        "## 效果",
        table(["模型", "命中/全量", "命中率", "误放", "误拒", "待复核", "执行失败", "正例通过", "坏例拒绝", "平衡命中率"],
              [[r["model"], f"{r['matched']}/{r['total']}", f"{r['match_rate']:.1%}",
                r["false_accept"], r["false_reject"], r["needs_review"], r["failed"],
                f"{r['positive_pass']}/{r['positive_total']}", f"{r['negative_reject']}/{r['negative_total']}",
                f"{r['balanced_match_rate']:.1%}" if r["balanced_match_rate"] is not None else "—"] for r in models]),
        "命中率 = 正确自动判定数 / 全部样本数；待复核和执行失败均计入分母且不算命中。"
        "误放指坏例被判通过，误拒指正例被判不通过。平衡命中率为正例通过率与坏例拒绝率的平均值。"
        "各模型仅运行一次，少量样本差异不能证明稳定优势。",
        table(["模型", "自动判定覆盖率", "已自动判定部分准确率", "全量命中率 Wilson 95% 区间"],
              [[r["model"], f"{r['coverage']:.1%}",
                f"{r['accuracy_on_completed']:.1%}" if r["accuracy_on_completed"] is not None else "—",
                "–".join(f"{value:.1%}" for value in r["match_rate_wilson95"])] for r in models]),
        "这些区间只是样本数对应的描述性二项区间；数据按已知问题收集，且同一任务内存在相关性，"
        "不代表随机生产流量上的泛化置信度。",
        "## 用量与费用",
        table(["模型", "HTTP 请求/重试", "输入 token", "输出 token", "缓存 token", "原价估算 ¥", "缓存折算已知部分 ¥", "每条原价 ¥", "用量未知请求"],
              [[r["model"], f"{r['calls']}/{r['retries']}", r["prompt_tokens"], r["completion_tokens"],
                r["cached_tokens"], number(r["list_cny"], 4), number(r["cache_adjusted_known_cny"], 4),
                number(r["mean_cny_per_episode"], 4), r["unknown_usage_calls"]] for r in models]),
        f"全部模型有回执可计算的原价合计 **¥{sum(r['list_cny'] for r in models):.4f}**。"
        "费用由真实 API usage × 2026-09-15 北京地域公开按量单价计算；包含失败结果和重试中的已知用量。"
        "未读取账户账单，未扣免费额度、账号优惠、Batch 或夜间折扣，因此这里是估算。"
        "缺失 usage 的请求费用未知，不能当作免费；有未知请求时表内金额仅覆盖已知用量。",
        f"缓存价格未知但回执含缓存 token 的请求共 {sum(r['unknown_cache_price_calls'] for r in models)} 次；"
        "这些请求不计入缓存折算已知金额，但原价金额仍保留完整输入。"
        "缓存命中受这次运行的并行顺序和共享前缀影响，主要横向比较使用原价列。",
        table(["模型", "任务调用/原价 ¥", "画质调用/原价 ¥", "成功返回结果 P50 秒", "P95 秒", "延迟样本数"],
              [[r["model"], f"{r['by_stage']['task']['calls']} / {r['by_stage']['task']['list_cny']:.4f}",
                f"{r['by_stage']['quality']['calls']} / {r['by_stage']['quality']['list_cny']:.4f}",
                number(r["api_p50_s"]), number(r["api_p95_s"]), r["valid_latency_n"]] for r in models]),
        "每条延迟是任务与画质 HTTP 请求耗时之和，包含重试 HTTP 耗时；仅统计完成业务判定或待复核的结果。"
        "失败结果不进入该延迟分布，数量见上表。限速等待、重试退避、媒体准备和排队不计入；"
        "每条墙钟耗时与限速等待另保存在 episodes.csv，因此此表不能直接当作 GUI 响应时间。",
        "## 分类型结果",
        table(["GT 类别 / 数量", *[r["model"] for r in models]],
              [[f"{reason} / {best['by_reason'][reason]['total']}",
                *[r["by_reason"][reason]["matched"] for r in models]] for reason in best["by_reason"]]),
        "单元格是该类别命中条数。拒绝标签正确不保证模型解释也正确；本轮没有单独标注解释与时间证据准确性。",
        table(["模型", "原 development 命中/数量", "原 holdout 命中/数量", "失败阶段"],
              [[r["model"], *[f"{r['by_split'][name]['matched']}/{r['by_split'][name]['total']}"
                              for name in ("development", "holdout")], json.dumps(r["failures_by_stage"], ensure_ascii=False)]
               for r in models]),
        "## 固定协议与局限",
        f"实验 `{frozen['sha256']}`；执行代码 Git `{frozen['git_commit']}`。"
        f"准备于 `{frozen['created_at']}`，最早请求 `{summary['started_at']}`，最晚结果 `{summary['finished_at']}`（UTC）。",
        f"数据为任务 DL-NA52UU 的 {best['total']} 条记录：{best['positive_total']} 正例、{best['negative_total']} 坏例，"
        "保留原 development 68 / holdout 30 分组。这批数据此前已用于问题排查与 prompt 开发，"
        "本轮属于固定已知数据回归，原 holdout 也不能再视作独立、未见过的测试集。",
        f"每秒采样并保留首尾，主镜头、左腕和右腕按时间对齐拼接。共 {sum(frames)} 张候选拼接帧，"
        f"每条 {min(frames)}–{max(frames)} 张。任务调用接收任务指令、规则、当前资源参考图、全部候选帧及夹爪/触觉时序；"
        "独立画质调用只接收同一套候选帧和画质 prompt。GT 在离线统计阶段关联，不进入两类模型请求。"
        "本次重新读取任务资源并核验与缓存一致，逐条验证 MCAP、图片、信号和输入哈希。",
        f"temperature={frozen['plan']['temperature']}，max_tokens={frozen['plan']['max_tokens']}，"
        f"非思考模式，单次 HTTP 超时 {frozen['plan']['timeout_s']} 秒，最多 {frozen['plan']['attempts']} 次 HTTP 尝试。"
        "每个模型一个串行 worker，不同模型并行；Qwen3-VL 两个快照按 100k TPM 的 80% 节奏限速，"
        "其余按 1M TPM 的 80% 限速。已完成结果原样复用，业务或格式失败不为提分而补跑。"
        "首条兼容性测试属于同一批结果及费用，没有额外重复收费。",
        "沿用当前适配器：Qwen3.8-Max 使用 strict JSON Schema，其余使用 JSON object。"
        "两种模式接收相同消息，但输出约束强度不同，所以本轮比较的是当前应用中的可用效果；"
        "不能把格式失败差异全部归因于模型视觉能力。",
        table(["请求模型 ID", "回执返回 model", "响应格式", "含非空 reasoning_content 的请求"],
              [[r["model"], ", ".join(r["returned_models"]), ", ".join(r["response_formats"]),
                r["reasoning_response_calls"]] for r in models]),
        "qwen3.8-flash 和 qwen-vl-max 使用服务别名；回执若仍返回别名，无法据此确定底层权重版本。"
        "HTTP 成功与业务成功分开统计；输出截断、额外字段、缺帧或无效证据等均保留失败。",
        "## 价格来源",
        "单位：人民币 / 百万 token。按每次请求的完整输入 token 数选择一个阶梯，"
        "该阶梯同时用于本次全部输入与输出，边界包含上限；缓存 token 属于输入的子集，不能重复累加。",
        table(["模型 / 官方来源", "输入上限", "输入", "输出", "缓存输入"],
              [[f"[{model}]({value['source']})", tier[0], tier[1], tier[2], "未采用" if tier[3] is None else tier[3]]
               for model, value in frozen["pricing"]["models"].items() for tier in value["tiers"]]),
        "## 复算与证据",
        "```bash\n# 新环境\nbash scripts/bootstrap.sh\n.tools/bin/just setup\n\n"
        "# 查看已保存结果；不调用模型\n.tools/bin/just benchmark-report\n\n"
        "# 从回执生成本报告；不调用模型\n.tools/bin/just benchmark-report --publish docs/benchmarks/2026-09-15\n\n"
        "# 有原始数据、任务资源网络和 Qwen 凭据时运行实验\n.tools/bin/just benchmark prepare\n"
        ".tools/bin/just benchmark run\n```",
        "运行配置见 [benchmark.toml](../../../config/benchmark.toml)，价格快照见 "
        "[cny_20260915.json](../../../config/prices/cny_20260915.json)。"
        "复用现有实验要求模型代码、prompt、规则、配置和数据哈希与冻结版本一致；"
        "另做一轮需使用新的配置文件与 output 目录，不能覆盖本轮回执。",
        "[summary.json](summary.json) 保存完整统计；[models.csv](models.csv)、[episodes.csv](episodes.csv)、"
        "[calls.csv](calls.csv) 分别记录模型汇总、每条 GT/预测/费用和每次请求用量；"
        "[protocol.json](protocol.json) 保存代码、配置与输入哈希。",
        f"原始脱敏请求、供应商响应及结果保存在 `{frozen['plan']['output']}/`，未将图片或运行目录提交到 Git。"
        f"发布前核验 {summary['audit']['requests_checked']} 份请求均与对应样本、阶段的冻结消息一致，"
        "并核对了结果范围和配置来源。",
    ]
    return "\n\n".join(sections) + "\n"


def publish(work, destination):
    summary, episodes, calls, frozen = summarize(work)
    if summary["state"] != "finished":
        raise ValueError("Benchmark is still running; use progress output until all models finish")
    summary["audit"] = audit_receipts(work, frozen)
    summary["started_at"] = min(row["started_at"] for row in calls if row["started_at"])
    summary["finished_at"] = max(read(path)["finished_at"] for path in (work / "models").glob("*/results/" + RESULT_GLOB))
    protocol = {key: frozen[key] for key in ("sha256", "created_at", "git_commit", "purpose", "plan", "plan_sha256",
                                            "runner_sha256", "configuration", "input_audit", "resources", "model_settings", "pricing")}
    documents = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                 "protocol.json": json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                 "models.csv": csv_text(summary["models"]), "episodes.csv": csv_text(episodes),
                 "calls.csv": csv_text(calls), "README.md": markdown(summary, frozen)}
    # Derived documents are deterministic; never silently overwrite differing published evidence.
    for name, content in documents.items():
        path = destination / name
        if path.exists() and path.read_text() != content:
            raise ValueError("Published document differs: " + str(path))
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in documents.items():
        (destination / name).write_text(content, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("artifacts/experiments/model_benchmark_20260915"))
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    summary = publish(args.work, args.publish) if args.publish else summarize(args.work)[0]
    print(json.dumps([{key: row[key] for key in ("model", "matched", "failed", "not_run", "calls", "list_cny")}
                     for row in summary["models"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
