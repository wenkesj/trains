import av
from PIL import Image

from trains.datum.adapters.video import _video_parts


def test_video_sampling_preserves_timestamps_and_frame_limit(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=2)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for color in ["red", "green", "blue", "white"]:
            frame = av.VideoFrame.from_image(Image.new("RGB", (16, 16), color))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    parts = _video_parts(str(path), fps=1, max_frames=2)
    assert len(parts) == 4
    assert parts[0]["text"] == "Frame at 0.000000s"
    assert parts[2]["text"] == "Frame at 1.000000s"
