"""MCAP integrity inspection and continuous H.264 decoding with original time."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
from mcap.reader import make_reader

from ..models import MediaError
from .flatbuffer import FlatbufferSchema, MessageView


GRIPPER_TOPICS = (
    "/observation/left_gripper/gripper/joint_position",
    "/observation/right_gripper/gripper/joint_position",
)


def inspect_mcap(mcap_path: Path, required_topics: tuple | list = ()) -> dict:
    """Read indexed channel counts; the streaming decoder verifies chunk CRCs."""
    try:
        with Path(mcap_path).open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            summary = reader.get_summary()
            if summary is None or summary.statistics is None:
                raise MediaError("MCAP summary/statistics are missing")
            stats = summary.statistics
            counts: dict[str, int] = {}
            for channel in summary.channels.values():
                counts[channel.topic] = counts.get(channel.topic, 0) + stats.channel_message_counts.get(
                    channel.id, 0)
            return {
                "source_channels": sorted(counts), "message_counts": counts,
                "missing_required_channels": sorted(set(required_topics) - counts.keys()),
                "empty_required_channels": sorted(topic for topic in required_topics
                                                   if topic in counts and counts[topic] == 0),
                "duration_s": (stats.message_end_time - stats.message_start_time) / 1e9,
                "start_time_ns": stats.message_start_time,
            }
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(f"Cannot inspect MCAP: {type(exc).__name__}") from exc


def _timestamp(view: MessageView, log_time: int) -> tuple[int, bool]:
    stamp = view.get("timestamp")
    if isinstance(stamp, MessageView):
        sec, nsec = stamp.get("sec"), stamp.get("nsec")
        if isinstance(sec, int) and isinstance(nsec, int) and 0 <= nsec < 1_000_000_000:
            return sec * 1_000_000_000 + nsec, False
    return log_time, True


def _messages(path: Path, topics: list | tuple):
    schemas = {}
    with Path(path).open("rb") as stream:
        for schema, channel, message in make_reader(stream, validate_crcs=True).iter_messages(
            topics=topics, log_time_order=True
        ):
            if schema is None or schema.encoding not in ("flatbuffer", "flatbuffers"):
                raise MediaError("Unsupported or missing MCAP FlatBuffers schema")
            if schema.id not in schemas:
                schemas[schema.id] = FlatbufferSchema(schema.data, schema.name)
            yield channel.topic, schemas[schema.id], message


def decode_video(path: Path, topic: str, start_ns: int, warnings: list[str]) -> Iterator[tuple[float, av.VideoFrame]]:
    """Decode all packets before sampling. B-frame PTS is kept by the decoder."""
    codec = av.CodecContext.create("h264", "r")
    previous = -1.0
    decoded_count = 0

    def timed(frame: av.VideoFrame) -> tuple[float, av.VideoFrame]:
        nonlocal previous, decoded_count
        if frame.pts is None:
            raise MediaError("Decoder did not preserve the source presentation timestamp")
        # PyAV flush frames may omit the time base; PTS keeps our nanosecond units.
        time_s = float(frame.pts * (frame.time_base or Fraction(1, 1_000_000_000)))
        if time_s < -0.1 or time_s <= previous:
            raise MediaError("Video timestamps are not strictly increasing within the episode")
        previous, decoded_count = time_s, decoded_count + 1
        return max(0.0, time_s), frame

    try:
        for _, schema, message in _messages(path, [topic]):
            if schema.root.name != "foxglove.CompressedVideo":
                raise MediaError("Unsupported video schema; expected foxglove.CompressedVideo")
            video = schema.decode(message.data)
            if video.get("format") != "h264":
                raise MediaError("Unsupported video codec; expected h264")
            data = video.get("data")
            if not isinstance(data, bytes) or not data:
                raise MediaError("Empty or malformed compressed video packet")
            timestamp_ns, fallback = _timestamp(video, message.log_time)
            if fallback and "video_timestamp_fallback_to_mcap_log_time" not in warnings:
                warnings.append("video_timestamp_fallback_to_mcap_log_time")
            packet = av.Packet(data)
            packet.pts = timestamp_ns - start_ns
            packet.time_base = Fraction(1, 1_000_000_000)
            for frame in codec.decode(packet):
                yield timed(frame)
        for frame in codec.decode(None):
            yield timed(frame)
        if decoded_count == 0:
            raise MediaError("No video frames could be decoded")
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(f"Cannot decode video: {type(exc).__name__}") from exc


def read_grippers(path: Path, start_ns: int) -> dict[str, list[tuple[float, float]]]:
    """Read positions using the embedded JointStates schema, including joints[]."""
    result = {topic: [] for topic in GRIPPER_TOPICS}
    try:
        for topic, schema, message in _messages(path, GRIPPER_TOPICS):
            if schema.root.name != "foxglove.JointStates":
                raise MediaError("Unsupported gripper schema; expected foxglove.JointStates")
            view = schema.decode(message.data)
            timestamp_ns, fallback = _timestamp(view, message.log_time)
            if fallback:
                raise MediaError("Gripper acquisition timestamps are missing")
            joints = view.get("joints")
            if not isinstance(joints, list) or len(joints) != 1 or not isinstance(joints[0], MessageView):
                raise MediaError("Expected one measured joint per gripper channel")
            value = joints[0].get("position")
            if not isinstance(value, (int, float)):
                raise MediaError("Gripper position is missing")
            result[topic].append(((timestamp_ns - start_ns) / 1e9, float(value)))
        return result
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(f"Cannot read gripper positions: {type(exc).__name__}") from exc
