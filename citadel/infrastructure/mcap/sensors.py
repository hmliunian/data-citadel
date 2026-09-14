"""Timestamped gripper position and relative tactile load; never infer a grasp label."""
from collections import defaultdict
import math
from pathlib import Path
import struct

from mcap.reader import make_reader

from ..files import fingerprint, read, write
from .flatbuffer import Schema

TOPICS = {
    **{side + "_position": "/observation/" + side + "_gripper/gripper/joint_position"
       for side in ("left", "right")},
    **{side + "_finger" + str(f): "/observation/tactile_" + side + "_gripper_finger" +
       str(f) + "/tactile/tactile_point_cloud2"
       for side in ("left", "right") for f in (0, 1)},
}


def read_gripper(path):
    value = read(path)
    if fingerprint({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]:
        raise ValueError("Cached gripper signals changed")
    return value


def prepare_gripper(source: Path, folder: Path, origin: int, source_hash: str):
    signature = {"version": "relative-gripper-v1", "mcap_sha256": source_hash, "origin_ns": origin}
    path = folder / "gripper.json"
    if path.exists():
        value = read_gripper(path)
        if value["signature"] != signature:
            raise ValueError("Gripper configuration changed; use a new run")
        return value
    channels = {name: {"topic": topic, "samples": []} for name, topic in TOPICS.items()}
    warnings, schemas = set(), {}
    by_topic = {topic: name for name, topic in TOPICS.items()}
    with source.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        for schema, channel, message in reader.iter_messages(topics=list(by_topic)):
            name = by_topic[channel.topic]
            try:
                expected = "foxglove.JointStates" if name.endswith("_position") else "discover.TactileData"
                if schema is None or schema.encoding not in ("flatbuffer", "flatbuffers"):
                    raise ValueError("Unsupported gripper encoding")
                if schema.id not in schemas:
                    schemas[schema.id] = Schema(schema.data, expected)
                if schema.name != expected:
                    raise ValueError("Gripper schema mismatch")
                data = schemas[schema.id].decode(message.data)
                stamp = data.get("timestamp") or {}
                if (isinstance(stamp.get("sec"), int) and stamp["sec"] > 0
                        and isinstance(stamp.get("nsec"), int) and 0 <= stamp["nsec"] < 10**9):
                    ns = stamp["sec"] * 10**9 + stamp["nsec"]
                else:
                    ns = message.log_time
                    warnings.add(name + ":timestamp_fallback")
                if abs(ns - message.log_time) > 100_000_000:
                    raise ValueError("Gripper clock differs from video clock")
                if name.endswith("_position"):
                    if len(data["joints"]) != 1:
                        raise ValueError("Expected one gripper position")
                    value = data["joints"][0]["position"]
                else:
                    if not data["points"]:
                        raise ValueError("Empty tactile sample")
                    value = sum(math.sqrt(sum(p[k] ** 2 for k in ("fx", "fy", "fz")))
                                for p in data["points"])
                if not math.isfinite(value):
                    raise ValueError("Nonfinite gripper sample")
                samples = channels[name]["samples"]
                stamp_s = (ns - origin) / 1e9
                if samples and stamp_s <= samples[-1][0]:
                    raise ValueError("Gripper timestamps must increase")
                samples.append([stamp_s, value])
            except (ValueError, KeyError, TypeError, IndexError, struct.error):
                warnings.add(name + ":invalid_sample")
    for name, channel in channels.items():
        samples = channel["samples"]
        if not samples:
            warnings.add(name + ":missing")
        elif any(b[0] - a[0] > 0.1 for a, b in zip(samples, samples[1:])):
            warnings.add(name + ":sample_gap")
    value = {"signature": signature, "channels": channels, "warnings": sorted(warnings)}
    value["sha256"] = fingerprint(value)
    write(path, value)
    return value


def intervals(gripper, frames):
    """Keep 0.1s extrema between adjacent visual frames; no interpolation or zero filling."""
    previous, output = 0, []
    for index, frame in enumerate(frames):
        end = frame["time_s"]
        buckets = defaultdict(lambda: [[] for _ in TOPICS])
        for i, name in enumerate(TOPICS):
            for stamp, value in gripper["channels"].get(name, {}).get("samples", []):
                if previous <= stamp <= end and (index == 0 or stamp > previous):
                    buckets[math.floor(stamp * 10)][i].append(value)
        output.append([
            [round(max(previous, b / 10), 6), round(min(end, (b + 1) / 10), 6)] +
            [[round(min(v), 2), round(max(v), 2)] if v else None for v in columns]
            for b, columns in sorted(buckets.items())])
        previous = end
    return output
