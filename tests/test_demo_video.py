"""Tests for the Alpamayo demo-video renderer."""

from pathlib import Path

import numpy as np
from PIL import Image

from alpamayo_r1.demo_video import (
    CameraVideo,
    DemoScenario,
    RenderConfig,
    ResourceSnapshot,
    StageLatenciesMs,
    render_frame,
)


def _make_scenario(num_samples: int = 3) -> DemoScenario:
    camera = CameraVideo(
        name="FRONT WIDE",
        frames=(
            Image.new("RGB", (320, 180), (40, 80, 120)),
            Image.new("RGB", (320, 180), (60, 100, 140)),
        ),
    )
    rng = np.random.default_rng(0)
    samples = tuple(
        (np.array([[0.0, 0.0], [0.5, 4.0]], dtype=np.float32) + rng.normal(scale=0.1, size=(2, 2)).astype(np.float32))
        for _ in range(num_samples)
    )
    return DemoScenario(
        clip_id="synthetic-clip-0123",
        reasoning="Keep a safe path through the open lane.",
        camera=camera,
        num_cameras=4,
        backbone_summary="Qwen3-VL-8B backbone + 2.3B flow-matching expert",
        history_xy_m=np.array([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0]], dtype=np.float32),
        ground_truth_xy_m=np.array([[0.0, 0.0], [0.2, 4.1]], dtype=np.float32),
        sample_xy_m=samples,
        best_index=0,
        stage_latencies_ms=StageLatenciesMs(vision=120.0, prefill=340.0, decode=610.0, diffusion=90.0),
        resources=ResourceSnapshot(cpu_percent=42.5, rss_mb=2048.0, num_threads=24, per_core_percent=(10.0, 55.0, 30.0, 80.0)),
    )


def test_render_frame_changes_when_trajectory_advances() -> None:
    """Render a complete dashboard whose trajectory advances with time."""
    # Given: a synthetic scenario with two distinct trajectory steps.
    scenario = _make_scenario()
    config = RenderConfig(width=960, height=540, fps=10)

    # When: the first and second timeline frames are rendered.
    first = render_frame(scenario, config, step=0)
    second = render_frame(scenario, config, step=1)

    # Then: both are valid frames and the advancing dashboard changes visually.
    assert first.size == (960, 540)
    assert first.mode == "RGB"
    assert first.tobytes() != second.tobytes()


def test_encode_demo_video_writes_playable_mp4(tmp_path: Path) -> None:
    """Encode rendered dashboard frames as an MP4 file."""
    # Given: the smallest complete synthetic scenario.
    scenario = _make_scenario(num_samples=2)
    output_path = tmp_path / "demo.mp4"

    # When: the scenario is encoded through the public video API.
    from alpamayo_r1.demo_video import encode_demo_video

    encode_demo_video(scenario, output_path, RenderConfig(width=640, height=360, fps=1))

    # Then: ffmpeg produced a non-empty MP4 container.
    assert output_path.read_bytes()[4:8] == b"ftyp"
