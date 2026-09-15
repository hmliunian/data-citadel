"""Pure signal aggregation shared by request preparation and media adapters."""
from collections import defaultdict
import math

CHANNELS = ("left_position", "right_position", "left_finger0", "left_finger1",
            "right_finger0", "right_finger1")


def intervals(gripper, frames):
    """Keep 0.1s extrema between adjacent visual frames; no interpolation or zero filling."""
    previous, output = 0, []
    for index, frame in enumerate(frames):
        end = frame["time_s"]
        buckets = defaultdict(lambda: [[] for _ in CHANNELS])
        for i, name in enumerate(CHANNELS):
            for stamp, value in gripper["channels"].get(name, {}).get("samples", []):
                if previous <= stamp <= end and (index == 0 or stamp > previous):
                    buckets[math.floor(stamp * 10)][i].append(value)
        output.append([
            [round(max(previous, b / 10), 6), round(min(end, (b + 1) / 10), 6)] +
            [[round(min(v), 2), round(max(v), 2)] if v else None for v in columns]
            for b, columns in sorted(buckets.items())])
        previous = end
    return output
