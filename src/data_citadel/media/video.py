"""Browser MP4 previews, independent of model input and source MCAP files."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile

import av

from ..models import Episode, MediaError
from .mcap_reader import decode_video, inspect_mcap


_TIME_BASE = Fraction(1, 1_000_000_000)


def export_video(episode: Episode, topic: str, cache_dir: Path) -> Path:
    """Encode one original-resolution camera, preserving its presentation gaps.

    The first camera frame becomes time zero. MP4 presentation timestamps use
    nanoseconds; frame indices or a fixed playback rate never replace them.
    Only fully closed MP4 files are atomically published as cache entries.
    """
    try:
        source = episode.mcap_path.resolve()
        stat = source.stat()
        fingerprint = json.dumps(["mcap-mp4-v1", str(source), stat.st_size, stat.st_mtime_ns, topic])
        cache_dir = Path(cache_dir)
        destination = cache_dir / (hashlib.sha256(fingerprint.encode()).hexdigest() + ".mp4")
        if destination.is_file() and destination.stat().st_size:
            return destination
        inspection = inspect_mcap(source, [topic])
        if inspection["missing_required_channels"] or inspection["empty_required_channels"]:
            raise MediaError("missing_camera_topic: requested video channel is missing or empty")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="preview-", suffix=".tmp", dir=cache_dir,
                                         delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with av.open(str(temporary), mode="w", format="mp4", options={"movflags": "+faststart"}) as output:
                stream = None
                first_time = None
                for _, decoded in decode_video(source, topic, inspection["start_time_ns"], []):
                    timestamp = decoded.pts * (decoded.time_base or _TIME_BASE)
                    if stream is None:
                        if timestamp < 0:
                            raise MediaError("Negative first video timestamp cannot match sampled frame seeking")
                        if decoded.width % 2 or decoded.height % 2:
                            raise MediaError("H264 yuv420p preview requires even source dimensions")
                        stream = output.add_stream("libx264")
                        stream.width, stream.height = decoded.width, decoded.height
                        stream.pix_fmt = "yuv420p"
                        stream.time_base = stream.codec_context.time_base = _TIME_BASE
                        stream.options = {"preset": "veryfast", "crf": "20", "bf": "0"}
                        first_time = timestamp
                    if (decoded.width, decoded.height) != (stream.width, stream.height):
                        raise MediaError("Source video resolution changes within the episode")
                    frame = decoded.reformat(format="yuv420p")
                    frame.pts = round((timestamp - first_time) / _TIME_BASE)
                    frame.time_base = _TIME_BASE
                    for packet in stream.encode(frame):
                        output.mux(packet)
                if stream is None:
                    raise MediaError("No video frames could be exported")
                for packet in stream.encode(None):
                    output.mux(packet)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
    except MediaError:
        raise
    except Exception as error:
        raise MediaError(f"Cannot export video preview: {type(error).__name__}") from error
