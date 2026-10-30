import io
import math
from typing import Any

import av
from PIL import Image

from trains.datasets.assets import read_asset
from trains.formats.records import Message


class VideoRenderer:
    """Expand videos into timestamped image frames before native model rendering."""

    def __init__(self, renderer: Any, *, fps: float, max_frames: int) -> None:
        """Configure deterministic frame extraction before native rendering.

        Args:
            renderer: Native renderer responsible for exact chat and media token layout.
            fps: Frame extraction rate in frames per second.
            max_frames: Maximum number of frames retained from one video.
        """
        self.renderer, self.fps, self.max_frames = renderer, fps, max_frames

    def build_supervised_examples(self, messages: list[Message]) -> Any:
        """Render complete messages while preserving the configured assistant masks."""
        prepared = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") in {"video", "input_video", "video_url"}:
                        uri = part.get("video", part.get("url", part.get("video_url")))
                        if isinstance(uri, dict):
                            uri = uri["url"]
                        parts.extend(_video_parts(uri, fps=self.fps, max_frames=self.max_frames))
                    else:
                        parts.append(part)
                content = parts
            prepared.append({**message, "content": content})
        return self.renderer.build_supervised_examples(prepared)


def _video_parts(uri: str, *, fps: float, max_frames: int) -> list[dict[str, Any]]:
    """Decode frames in timestamp order using an explicit deterministic sampling policy."""
    if not math.isfinite(fps) or fps <= 0 or type(max_frames) is not int or max_frames <= 0:
        raise ValueError("Video fps and max_frames must be positive")
    data, _ = read_asset(uri)
    parts, next_time = [], 0.0
    with av.open(io.BytesIO(data), mode="r") as container:
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError("Video frames require timestamps")
            if frame.time < next_time:
                continue
            image = io.BytesIO()
            frame.to_image().save(image, format="PNG")
            import_value = image.getvalue()
            parts.extend(
                [
                    {"type": "text", "text": f"Frame at {frame.time:.6f}s"},
                    {"type": "image", "image": Image.open(io.BytesIO(import_value)).copy()},
                ]
            )
            next_time = frame.time + 1.0 / fps
            if len(parts) // 2 >= max_frames:
                break
    if not parts:
        raise ValueError("Video contains no decodable frames")
    return parts
