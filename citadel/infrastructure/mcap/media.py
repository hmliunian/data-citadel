"""Decode actual camera times, align at 1s and stitch three complete views."""
from __future__ import annotations

import bisect
from fractions import Fraction
from pathlib import Path

import av
from mcap.reader import make_reader
from PIL import Image, ImageDraw, ImageFont

from ..files import file_hash, read, write
from .flatbuffer import Schema
from .sensors import prepare_gripper

TOPICS = {
    "main": "/camera/coracam_head/left_h264/video",
    "left_wrist": "/camera/coracam_lefthand/left_h264/video",
    "right_wrist": "/camera/coracam_righthand/left_h264/video",
}
VERSION = "aligned-three-view-v1"


def decode(source: Path, topics: dict):
    cameras = {topic: view for view, topic in topics.items()}
    streams = {view: [] for view in topics}
    codecs = {view: av.CodecContext.create("h264", "r") for view in topics}
    warnings, schemas = [], {}
    with source.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError("MCAP summary is required")
        origin = summary.statistics.message_start_time
        for schema, channel, message in reader.iter_messages(topics=list(cameras)):
            if schema is None or schema.encoding not in ("flatbuffer", "flatbuffers"):
                raise ValueError("Unsupported camera encoding")
            if schema.id not in schemas:
                schemas[schema.id] = Schema(schema.data)
            video = schemas[schema.id].decode(message.data)
            if video["format"] != "h264":
                raise ValueError("Unsupported video codec")
            stamp = video.get("timestamp") or {}
            if (isinstance(stamp.get("sec"), int) and stamp["sec"] > 0
                    and isinstance(stamp.get("nsec"), int) and 0 <= stamp["nsec"] < 10**9):
                ns = stamp["sec"] * 10**9 + stamp["nsec"]
            else:
                ns = message.log_time
                warnings.append("timestamp_fallback")
            view = cameras[channel.topic]
            packet = av.Packet(video["data"])
            packet.pts, packet.time_base = ns - origin, Fraction(1, 10**9)
            for frame in codecs[view].decode(packet):
                if frame.pts is None:
                    raise ValueError("Decoded camera frame has no timestamp")
                streams[view].append((origin + frame.pts, frame))
        for view, codec in codecs.items():
            if streams[view]:
                for frame in codec.decode(None):
                    if frame.pts is None:
                        raise ValueError("Flushed camera frame has no timestamp")
                    streams[view].append((origin + frame.pts, frame))
    for view, frames in streams.items():
        if any(b[0] <= a[0] for a, b in zip(frames, frames[1:])):
            raise ValueError("Camera timestamps must increase")
        if not frames:
            warnings.append(view + ":missing_camera")
    return origin, streams, sorted(set(warnings))


def nearest(times, target, tolerance):
    position = bisect.bisect_left(times, target)
    indices = [i for i in (position - 1, position) if 0 <= i < len(times)]
    if not indices:
        return None
    chosen = min(indices, key=lambda i: (abs(times[i] - target), i))
    return chosen if abs(times[chosen] - target) <= tolerance else None


def save_video(frames, path: Path):
    first = frames[0][0]
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height = frames[0][1].width, frames[0][1].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "20", "bf": "0"}
        stream.time_base = stream.codec_context.time_base = Fraction(1, 10**6)
        for ns, frame in frames:
            frame.pts, frame.time_base = round((ns - first) / 1000), Fraction(1, 10**6)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def render(work, episode_id, origin, streams, signature, warnings):
    folder = work / "media" / episode_id
    folder.mkdir(parents=True, exist_ok=True)
    active = {view: values for view, values in streams.items() if values}
    if not active:
        raise ValueError("No usable camera frames")
    interval = round(signature["interval_s"] * 10**9)
    tolerance = round(signature["tolerance_s"] * 10**9)
    if interval <= 0 or tolerance < 0:
        raise ValueError("Invalid sampling configuration")
    start = min(values[0][0] for values in active.values())
    end = max(values[-1][0] for values in active.values())
    endpoints = {v[i][0] for v in active.values() for i in (0, -1)}
    targets = sorted(endpoints | set(range(start, end + 1, interval)))
    # Very close camera endpoints share one target but retain each panel's real source time.
    targets = [target for i, target in enumerate(targets)
               if i == len(targets) - 1 or targets[i + 1] - target >= 1000]
    width = max(values[0][1].width for values in active.values())
    height = max(values[0][1].height for values in active.values())
    width += width % 2
    height += height % 2
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    result = {"episode_id": episode_id, "signature": signature, "origin_ns": origin,
              "frames": [], "videos": {}, "warnings": warnings,
              "incomplete": len(active) != 3 or bool(warnings)}
    times = {view: [ns for ns, _ in values] for view, values in streams.items()}
    stitched = []
    for index, target in enumerate(targets):
        frame_id = f"V{index:03d}"
        image = Image.new("RGB", (width * 3, height + 48), "#121c24")
        draw = ImageDraw.Draw(image)
        draw.text((8, 2), f"{frame_id} | time {(target - origin) / 1e9:.6f}s",
                  font=font, fill="white")
        sources = {}
        for column, view in enumerate(TOPICS):
            selected = nearest(times.get(view, []), target, tolerance)
            item = None
            if selected is not None:
                ns, frame = streams[view][selected]
                image.paste(frame.to_image(), (column * width, 48))
                item = {"topic": signature["topics"][view], "source_ns": ns,
                        "time_s": (ns - origin) / 1e9,
                        "video_time_s": (ns - streams[view][0][0]) / 1e9}
            elif target not in endpoints:
                result["incomplete"] = True
                result["warnings"].append(f"{frame_id}:{view}:no_aligned_frame")
            sources[view] = item
            label = view.upper() + (f' {item["time_s"]:.6f}s' if item else " NO FRAME")
            draw.text((column * width + 8, 26), label, font=font, fill="white")
        path = folder / (frame_id + ".jpg")
        image.save(path, quality=95)
        result["frames"].append({
            "frame_id": frame_id, "time_s": (target - origin) / 1e9,
            "video_time_s": (target - targets[0]) / 1e9, "endpoint": target in endpoints,
            "sources": sources, "path": str(path.relative_to(work)), "sha256": file_hash(path)})
        stitched.append((target, av.VideoFrame.from_image(image)))
    for view, frames in {**active, "stitched": stitched}.items():
        path = folder / (view + ".mp4")
        save_video(frames, path)
        result["videos"][view] = {
            "path": str(path.relative_to(work)), "sha256": file_hash(path),
            "start_s": (frames[0][0] - origin) / 1e9, "end_s": (frames[-1][0] - origin) / 1e9,
            "frame_count": len(frames)}
    write(folder / "media.json", result)
    return result


def prepare_media(work: Path, episode: dict, sampling: dict, topics=None):
    source = Path(episode["mcap_path"])
    actual = file_hash(source)
    if actual != episode["mcap_sha256"]:
        raise ValueError("Source MCAP changed after preparation")
    signature = {"version": VERSION, "mcap_sha256": actual,
                 "topics": topics or TOPICS, **sampling}
    cache = work / "media" / episode["episode_id"] / "media.json"
    if cache.exists():
        media = read(cache)
        if media["signature"] != signature:
            raise ValueError("Sampling changed; use a new run")
        for asset in media["frames"] + list(media["videos"].values()):
            path = (work / asset["path"]).resolve()
            if not path.is_relative_to(work.resolve()) or file_hash(path) != asset["sha256"]:
                raise ValueError("Cached media changed")
    else:
        origin, streams, warnings = decode(source, signature["topics"])
        media = render(work, episode["episode_id"], origin, streams, signature, warnings)
    return {**media, "gripper": prepare_gripper(source, cache.parent, media["origin_ns"], actual)}
