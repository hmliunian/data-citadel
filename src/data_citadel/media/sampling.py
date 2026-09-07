"""Budgeted frame selection with dense, low-resolution motion evidence."""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile

import numpy as np

from ..models import Episode, Frame, MediaError, SampledVideo
from .mcap_reader import decode_video, inspect_mcap, read_grippers


def gripper_event_times(samples: list[tuple[float, float]]) -> list[float]:
    """Detect open/close starts and ends in either direction.

    Five-point median smoothing removes isolated sensor spikes. Motion must have
    velocity above 15% of the robust position range per second for >=80 ms, and
    move >=3% of that range. The derivative uses a >=100 ms window. These are
    sampling heuristics, not evidence that a manipulation succeeded.
    """
    if len(samples) < 7:
        return []
    times, values = np.asarray(samples, dtype=float).T
    if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise MediaError("Invalid gripper position/time sequence")
    span = float(np.percentile(values, 95) - np.percentile(values, 5))
    if span < 1e-4:
        return []
    padded = np.pad(values, 2, mode="edge")
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    window = max(2, int(round(0.1 / float(np.median(np.diff(times))))))
    left = np.maximum(0, np.arange(len(times)) - window // 2)
    right = np.minimum(len(times) - 1, left + window)
    velocity = (smooth[right] - smooth[left]) / (times[right] - times[left])
    states = np.where(abs(velocity) >= max(span * 0.15, 1e-3), np.sign(velocity), 0)
    events: list[float] = []
    boundaries = np.r_[0, np.flatnonzero(np.diff(states)) + 1, len(states)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        end = stop - 1
        if (states[start] and times[end] - times[start] >= 0.08
                and abs(smooth[end] - smooth[start]) >= span * 0.03):
            events.extend((float(times[start]), float(times[end])))
    return sorted(set(events))


class _Motion:
    """Every decoded frame contributes, including frames not sent to the VLM."""

    threshold = 0.004  # Mean absolute grayscale difference, on a [0, 1] scale.

    def __init__(self):
        self.previous = None
        self.first_time = self.previous_time = self.stationary_start = None
        self.max_gap = 0.0
        self.count = 0
        self.intervals = []

    def add(self, timestamp_s: float, frame) -> None:
        gray = frame.reformat(width=96, height=54, format="gray").to_ndarray().astype(np.float32) / 255
        self.count += 1
        if self.previous is not None:
            gap = timestamp_s - self.previous_time
            self.max_gap = max(self.max_gap, gap)
            score = float(np.abs(gray - self.previous).mean())
            if score <= self.threshold and gap <= 0.25:
                if self.stationary_start is None:
                    self.stationary_start = self.previous_time
            else:
                self._close()
        else:
            self.first_time = timestamp_s
        self.previous, self.previous_time = gray, timestamp_s

    def _close(self) -> None:
        if self.stationary_start is not None:
            duration = self.previous_time - self.stationary_start
            if duration >= 5:
                self.intervals.append({"start_s": self.stationary_start,
                                       "end_s": self.previous_time, "duration_s": duration})
            self.stationary_start = None

    def result(self) -> dict:
        self._close()
        return {"analyzed_frames": self.count,
                "coverage_s": self.previous_time - self.first_time if self.count else 0,
                "max_gap_s": self.max_gap, "threshold_mean_abs_diff": self.threshold,
                "stationary_intervals": self.intervals,
                "max_stationary_duration_s": max((item["duration_s"] for item in self.intervals), default=0),
                "interpretation": "Low image motion is a candidate interval, not proof of invalid task execution."}


class VideoSampler:
    def __init__(self, camera_topic: str, max_frames: int = 96, max_image_size: int = 768,
                 cache_dir: Path | None = None):
        if max_frames < 2 or max_image_size < 32:
            raise ValueError("Need at least two frames and images of at least 32 pixels")
        self.camera_topic, self.max_frames, self.max_image_size = camera_topic, max_frames, max_image_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def _frame(self, time_s: float, decoded) -> Frame:
        image = decoded.to_image()
        image.thumbnail((self.max_image_size, self.max_image_size))
        output = BytesIO()
        image.save(output, format="JPEG", quality=85)
        return Frame(time_s, output.getvalue())

    def _cache_path(self, episode: Episode, strategy: str, interval_s: float) -> Path | None:
        if self.cache_dir is None:
            return None
        stat = episode.mcap_path.stat()
        key = json.dumps(["sampler-v1", str(episode.mcap_path.resolve()), stat.st_size,
                          stat.st_mtime_ns, self.camera_topic, self.max_frames,
                          self.max_image_size, strategy, interval_s])
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def sample(self, episode: Episode, strategy: str = "uniform", interval_s: float = 2.0) -> SampledVideo:
        if strategy not in ("uniform", "keyframes") or not np.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Expected uniform/keyframes and a positive finite interval")
        cache = self._cache_path(episode, strategy, interval_s)
        if cache is not None and cache.is_file():
            try:
                content = json.loads(cache.read_text())
                content["frames"] = [Frame(item["timestamp_s"], base64.b64decode(item["jpeg"], validate=True))
                                     for item in content["frames"]]
                return SampledVideo(**content)
            except (ValueError, KeyError, TypeError):
                pass  # A partial/obsolete cache is regenerated from the source.
        inspection = inspect_mcap(episode.mcap_path, [self.camera_topic])
        if inspection["missing_required_channels"] or inspection["empty_required_channels"]:
            raise MediaError("missing_camera_topic: configured video channel is missing or empty")
        warnings: list[str] = []
        duration = inspection["duration_s"]
        start_ns = inspection["start_time_ns"]
        targets = []
        if strategy == "keyframes":
            try:
                grippers = read_grippers(episode.mcap_path, start_ns)
                if any(not samples for samples in grippers.values()):
                    warnings.append("keyframe_gripper_channel_missing")
                targets = sorted(set(time for samples in grippers.values()
                                     for time in gripper_event_times(samples) if 0 <= time <= duration))
            except MediaError:
                warnings.append("keyframe_gripper_data_unusable")
            if not targets:
                warnings.append("keyframe_fallback_to_uniform_no_gripper_events")
        if strategy == "uniform" or not targets:
            # Compute count first to avoid an unbounded arange on invalid/extreme metadata.
            count = max(0, int(np.ceil(duration / interval_s)) - 1)
            if count > self.max_frames - 2:
                warnings.append("frame_budget_reduced_temporal_coverage")
                targets = list(np.linspace(0, duration, self.max_frames)[1:-1])
            else:
                targets = [index * interval_s for index in range(1, count + 1)]
        if len(targets) > self.max_frames - 2:
            indices = np.linspace(0, len(targets) - 1, self.max_frames - 2, dtype=int)
            targets = [targets[index] for index in indices]
            warnings.append("frame_budget_reduced_temporal_coverage")
        frames: dict[float, Frame] = {}
        target_index = 0
        previous = None
        motion = _Motion()
        for timestamp, decoded in decode_video(episode.mcap_path, self.camera_topic, start_ns, warnings):
            motion.add(timestamp, decoded)
            if previous is None:
                frames[timestamp] = self._frame(timestamp, decoded)
            while target_index < len(targets) and timestamp >= targets[target_index]:
                nearest = (timestamp, decoded)
                if previous is not None and abs(previous[0] - targets[target_index]) < abs(timestamp - targets[target_index]):
                    nearest = previous
                if nearest[0] not in frames:
                    frames[nearest[0]] = self._frame(*nearest)
                target_index += 1
            previous = (timestamp, decoded)
        if previous is None:
            raise MediaError("No video frames could be sampled")
        frames[previous[0]] = self._frame(*previous)
        duration = max(duration, previous[0])  # Delayed presentation may outlast MCAP log time.
        motion_result = motion.result()
        motion_result["source_channels"] = inspection["source_channels"]
        if motion_result["max_gap_s"] > 0.25:
            warnings.append("video_frame_gap_exceeds_250ms")
        if min(frames) > 0.5 or duration - max(frames) > 0.5:
            warnings.append("camera_does_not_cover_episode_boundaries")
        sampled = SampledVideo([frames[key] for key in sorted(frames)], duration,
                               self.camera_topic, strategy, list(dict.fromkeys(warnings)), motion_result)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            content = asdict(sampled)
            content["frames"] = [{"timestamp_s": frame.timestamp_s,
                                  "jpeg": base64.b64encode(frame.jpeg).decode()} for frame in sampled.frames]
            with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, suffix=".tmp", delete=False) as temporary:
                json.dump(content, temporary, ensure_ascii=False)
            Path(temporary.name).replace(cache)
        return sampled
