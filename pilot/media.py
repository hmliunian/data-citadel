"""Decode three source camera streams and keep image/video time mappings."""
from __future__ import annotations

import bisect
import math
from fractions import Fraction
from pathlib import Path

import av
from mcap.reader import make_reader

from .data import read_json, sha256, write_json
from .flatbuffer import FlatbufferSchema, MediaError, MessageView

TOPICS = {
    "main": "/camera/coracam_head/left_h264/video",
    "left_wrist": "/camera/coracam_lefthand/left_h264/video",
    "right_wrist": "/camera/coracam_righthand/left_h264/video",
}
SAMPLER_VERSION = "uniform-three-view-v1"


def sample_indices(times: list[float], interval: float = 2.0, tolerance: float = 0.1):
    if not times or interval <= 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Expected a non-empty increasing frame timeline")
    chosen, gaps = {0, len(times) - 1}, []
    for grid in range(math.ceil(times[0] / interval), math.floor(times[-1] / interval) + 1):
        target = grid * interval
        right = bisect.bisect_left(times, target)
        candidates = [i for i in (right - 1, right) if 0 <= i < len(times)]
        index = min(candidates, key=lambda i: (abs(times[i] - target), i))
        if abs(times[index] - target) <= tolerance:
            chosen.add(index)
        else:
            gaps.append(target)
    return sorted(chosen), gaps


def decode(path: Path, topic: str, start_ns: int, warnings: list[str]):
    codec, schemas = av.CodecContext.create("h264", "r"), {}
    previous = -1
    with path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        for schema, channel, message in reader.iter_messages(topics=[topic], log_time_order=True):
            if (schema is None or schema.name != "foxglove.CompressedVideo"
                    or schema.encoding not in ("flatbuffer", "flatbuffers")):
                raise MediaError("Unsupported MCAP video schema")
            if schema.id not in schemas:
                schemas[schema.id] = FlatbufferSchema(schema.data, schema.name)
            video = schemas[schema.id].decode(message.data)
            if video.get("format") != "h264":
                raise MediaError("Expected H.264 source video")
            stamp = video.get("timestamp")
            if isinstance(stamp, MessageView):
                sec, nsec = stamp.get("sec"), stamp.get("nsec")
            else:
                sec, nsec = None, None
            if isinstance(sec, int) and sec > 0 and isinstance(nsec, int) and 0 <= nsec < 10**9:
                source_ns = sec * 10**9 + nsec
            else:
                source_ns = message.log_time
                if "timestamp_fallback_to_log_time" not in warnings:
                    warnings.append("timestamp_fallback_to_log_time")
            packet = av.Packet(video.get("data"))
            packet.pts, packet.time_base = source_ns - start_ns, Fraction(1, 10**9)
            for frame in codec.decode(packet):
                if frame.pts is None or frame.pts <= previous:
                    raise MediaError("Non-increasing or missing decoded timestamp")
                previous = frame.pts
                yield frame.pts / 10**9, start_ns + frame.pts, frame
        for frame in codec.decode(None):
            if frame.pts is None or frame.pts <= previous:
                raise MediaError("Non-increasing or missing flush timestamp")
            previous = frame.pts
            yield frame.pts / 10**9, start_ns + frame.pts, frame


def export_video(frames, path: Path):
    first = frames[0][0]
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height = frames[0][2].width, frames[0][2].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "23", "bf": "0"}
        stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
        for time_s, _, frame in frames:
            frame.pts, frame.time_base = round((time_s - first) * 90000), Fraction(1, 90000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def prepare_media(run_dir: Path, episode: dict, sampling: dict):
    source = Path(episode["mcap_path"])
    if sha256(source) != episode["mcap_sha256"]:
        raise MediaError("Source MCAP no longer matches the verified snapshot")
    folder = run_dir / "media" / episode["episode_id"]
    cache_path = folder / "frames.json"
    signature = {"mcap_sha256": episode["mcap_sha256"], "sampler": SAMPLER_VERSION, **sampling}
    if cache_path.exists():
        saved = read_json(cache_path)
        if saved["signature"] != signature:
            raise MediaError("Media cache configuration changed; use a new pilot directory")
        for item in saved["frames"]:
            if sha256(run_dir / item["path"]) != item["sha256"]:
                raise MediaError("Cached image changed")
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as file:
        summary = make_reader(file, validate_crcs=True).get_summary()
    if summary is None or summary.statistics is None:
        raise MediaError("Missing MCAP summary")
    start_ns = summary.statistics.message_start_time
    available = {c.topic for c in summary.channels.values()}
    result = {"signature": signature, "episode_id": episode["episode_id"], "episode_start_ns": start_ns,
              "frames": [], "views": {}, "warnings": []}
    for view, topic in TOPICS.items():
        if topic not in available:
            result["warnings"].append(view + ":missing_camera")
            continue
        warnings = []
        frames = list(decode(source, topic, start_ns, warnings))
        if not frames:
            result["warnings"].append(view + ":empty_camera")
            continue
        times = [entry[0] for entry in frames]
        indices, gaps = sample_indices(times, sampling["interval_s"], sampling["interior_tolerance_s"])
        if gaps:
            warnings.append("interior_sampling_gap")
        for index in indices:
            time_s, source_ns, frame = frames[index]
            image_path = folder / f"{view}-{index:05d}.jpg"
            frame.to_image().save(image_path, quality=90)
            result["frames"].append({
                "frame_id": f"{view}-{index:05d}", "view": view, "topic": topic,
                "time_s": time_s, "source_ns": source_ns, "video_time_s": time_s - times[0],
                "path": str(image_path.relative_to(run_dir)), "sha256": sha256(image_path),
                "width": frame.width, "height": frame.height,
            })
        video_path = folder / (view + ".mp4")
        export_video(frames, video_path)
        result["views"][view] = {
            "topic": topic, "video_path": str(video_path.relative_to(run_dir)),
            "start_s": times[0], "end_s": times[-1], "decoded_frames": len(frames),
            "sampled_frames": len(indices), "uncovered_targets_s": gaps,
        }
        result["warnings"].extend(view + ":" + item for item in warnings)
        del frames
    result["frames"].sort(key=lambda item: (item["time_s"], item["view"]))
    write_json(cache_path, result)
    return result
