"""Render an Alpamayo trajectory prediction as a synchronized demo video."""

from __future__ import annotations

import copy
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps

FloatArray: TypeAlias = NDArray[np.float32]

BACKGROUND_TOP: Final = (10, 40, 102)
BACKGROUND_BOTTOM: Final = (28, 96, 196)
PANEL: Final = "#FFFFFF"
PANEL_BORDER: Final = "#D7DEEA"
GRID_LINE: Final = "#E4E9F2"
INK: Final = "#10182B"
MUTED: Final = "#5B6472"
WHITE: Final = "#F8FAFC"
ON_WHITE_ACCENT: Final = "#1462C4"
PREDICTED: Final = "#17A2E0"
PREDICTED_FAINT: Final = "#AED9F2"
GROUND_TRUTH: Final = "#E8536B"
HISTORY: Final = "#8C98A8"
EGO: Final = "#F2C94C"
BRAND_BG: Final = "#F8FAFC"
BRAND_BLUE: Final = "#0C2F6B"
STAGE_COLORS: Final = {
    "vision": "#2563EB",
    "prefill": "#0D9488",
    "decode": "#EA580C",
    "diffusion": "#E8536B",
    "total": "#7C3AED",
}
DEFAULT_CLIP_ID: Final = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_OUTPUT: Final = Path("outputs/alpamayo_demo_030c760c.mp4")
DEFAULT_NUM_SAMPLES: Final = 4


@dataclass(frozen=True, slots=True)
class CameraVideo:
    """A named, time-ordered camera stream."""

    name: str
    frames: tuple[Image.Image, ...]


@dataclass(frozen=True, slots=True)
class StageLatenciesMs:
    """Wall-clock time spent inside each model submodule during one rollout."""

    vision: float
    prefill: float
    decode: float
    diffusion: float

    @property
    def total(self) -> float:
        return self.vision + self.prefill + self.decode + self.diffusion


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Host-process resource usage sampled while the rollout ran."""

    cpu_percent: float
    rss_mb: float
    num_threads: int
    per_core_percent: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DemoScenario:
    """All model and sensor results needed by the video renderer."""

    clip_id: str
    reasoning: str
    camera: CameraVideo
    num_cameras: int
    backbone_summary: str
    history_xy_m: FloatArray
    ground_truth_xy_m: FloatArray
    sample_xy_m: tuple[FloatArray, ...]
    best_index: int
    stage_latencies_ms: StageLatenciesMs
    per_sample_latency_ms: tuple[float, ...]
    resources: ResourceSnapshot


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Pixel and timing configuration for an encoded demo."""

    width: int = 1920
    height: int = 1080
    fps: int = 10
    redact_camera: bool = False


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, max(size, 8))


def _gradient_background(config: RenderConfig) -> Image.Image:
    """A diagonal navy-to-blue gradient, evoking a tech/telemetry dashboard backdrop."""
    top = np.array(BACKGROUND_TOP, dtype=np.float32)
    bottom = np.array(BACKGROUND_BOTTOM, dtype=np.float32)
    diag = np.linspace(0.0, 1.0, config.width)[None, :] * 0.5 + np.linspace(0.0, 1.0, config.height)[:, None] * 0.5
    gradient = top[None, None, :] + diag[:, :, None] * (bottom - top)[None, None, :]
    return Image.fromarray(gradient.astype(np.uint8), mode="RGB")


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, radius: int, fill: str = PANEL) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=PANEL_BORDER, width=2)


def _section_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, scale: float) -> None:
    draw.text(xy, text, font=_font(round(19 * scale), bold=True), fill=WHITE)


def _flatten_strings(value: object) -> list[str]:
    """Flatten arbitrarily nested lists of strings, e.g. extra["cot"][batch]."""
    if isinstance(value, str):
        return [value]
    flattened: list[str] = []
    for item in value:
        flattened.extend(_flatten_strings(item))
    return flattened


def _xy_bounds(*arrays: FloatArray) -> tuple[float, float, float, float]:
    combined = np.concatenate(arrays, axis=0)
    lateral = -combined[:, 1]
    forward = combined[:, 0]
    x_margin = max(float(np.ptp(lateral)) * 0.12, 1.5)
    y_margin = max(float(np.ptp(forward)) * 0.08, 2.0)
    return (
        float(lateral.min()) - x_margin,
        float(lateral.max()) + x_margin,
        min(float(forward.min()) - y_margin, -1.0),
        float(forward.max()) + y_margin,
    )


def _traj_to_px(
    xy_m: FloatArray,
    bounds: tuple[float, float, float, float],
    box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    min_x, max_x, min_y, max_y = bounds
    left, top, right, bottom = box
    lateral = -xy_m[:, 1]
    forward = xy_m[:, 0]
    return [
        (
            round(left + (x - min_x) / (max_x - min_x) * (right - left)),
            round(bottom - (y - min_y) / (max_y - min_y) * (bottom - top)),
        )
        for x, y in zip(lateral, forward, strict=True)
    ]


def _navigation_summary(predicted_xy: FloatArray) -> str:
    """A short "continue <direction> for <distance>" readout computed from the real predicted path.

    Alpamayo-R1 does not take a navigation-route input or emit navigation text (that is an
    alpamayo1.5 feature) -- this is derived purely from the geometry of the predicted trajectory,
    not a model-generated instruction.
    """
    lateral = -predicted_xy[:, 1]
    net_lateral = float(lateral[-1] - lateral[0])
    distance_m = float(np.linalg.norm(np.diff(predicted_xy, axis=0), axis=1).sum())
    if net_lateral > 1.0:
        direction = "right"
    elif net_lateral < -1.0:
        direction = "left"
    else:
        direction = "straight"
    return f"Continue {direction} for {distance_m:.0f}m"


def _draw_legend_row(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    entries: tuple[tuple[str, str], ...],
    scale: float,
) -> None:
    x, y = xy
    swatch = round(10 * scale)
    font = _font(round(9.5 * scale))
    for label, color in entries:
        draw.line((x, y + swatch // 2, x + swatch, y + swatch // 2), fill=color, width=max(round(2.2 * scale), 2))
        draw.text((x + swatch + round(4 * scale), y), label, font=font, fill=MUTED)
        x += swatch + round(4 * scale) + round(draw.textlength(label, font=font)) + round(8 * scale)


def _draw_traj_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    bounds: tuple[float, float, float, float],
    history_xy: FloatArray,
    ground_truth_xy: FloatArray,
    other_paths: tuple[FloatArray, ...],
    best_xy: FloatArray,
    step: int,
    scale: float,
) -> None:
    left, top, right, bottom = box
    padding = round(16 * scale)
    legend_row_h = round(18 * scale)
    legend_h = legend_row_h * 2
    plot = (left + padding + round(26 * scale), top + round(34 * scale), right - padding, bottom - round(22 * scale) - legend_h)
    tick_font = _font(round(10 * scale))
    min_x, max_x, min_y, max_y = bounds
    for index in range(5):
        x = round(plot[0] + index * (plot[2] - plot[0]) / 4)
        y = round(plot[1] + index * (plot[3] - plot[1]) / 4)
        draw.line((x, plot[1], x, plot[3]), fill=GRID_LINE, width=1)
        draw.line((plot[0], y, plot[2], y), fill=GRID_LINE, width=1)
        x_value = min_x + index * (max_x - min_x) / 4
        y_value = max_y - index * (max_y - min_y) / 4
        draw.text((x, plot[3] + round(3 * scale)), f"{x_value:.0f}", font=tick_font, fill=MUTED, anchor="ma")
        draw.text((plot[0] - round(6 * scale), y), f"{y_value:.0f}", font=tick_font, fill=MUTED, anchor="rm")
    draw.text((left + padding, top + round(6 * scale)), title, font=_font(round(15 * scale), bold=True), fill=ON_WHITE_ACCENT)

    end = min(step + 1, len(ground_truth_xy))
    line_width = max(round(2 * scale), 1)
    for other in other_paths:
        pixels = _traj_to_px(other[:end], bounds, plot)
        if len(pixels) > 1:
            draw.line(pixels, fill=PREDICTED_FAINT, width=line_width, joint="curve")
    history_px = _traj_to_px(history_xy, bounds, plot)
    truth_px = _traj_to_px(ground_truth_xy[:end], bounds, plot)
    best_px = _traj_to_px(best_xy[:end], bounds, plot)
    if len(history_px) > 1:
        draw.line(history_px, fill=HISTORY, width=line_width, joint="curve")
    if len(truth_px) > 1:
        draw.line(truth_px, fill=GROUND_TRUTH, width=line_width + 1, joint="curve")
    if len(best_px) > 1:
        draw.line(best_px, fill=PREDICTED, width=line_width + 1, joint="curve")

    ego_px = history_px[-1] if history_px else _traj_to_px(np.zeros((1, 2), dtype=np.float32), bounds, plot)[0]
    ego_size = max(round(5 * scale), 3)
    draw.rectangle(
        (ego_px[0] - ego_size, ego_px[1] - ego_size * 2, ego_px[0] + ego_size, ego_px[1] + ego_size),
        fill=EGO,
        outline=INK,
        width=1,
    )

    primary_legend = (("History", HISTORY), ("Ground Truth", GROUND_TRUTH), ("Prediction", PREDICTED), ("Ego Vehicle", EGO))
    legend_top = bottom - legend_h + round(2 * scale)
    _draw_legend_row(draw, (left + padding, legend_top), primary_legend, scale)
    if other_paths:
        _draw_legend_row(draw, (left + padding, legend_top + legend_row_h), (("Other Predictions", PREDICTED_FAINT),), scale)


def _overlay_box(canvas: Image.Image, box: tuple[int, int, int, int], opacity: float) -> None:
    region = canvas.crop(box)
    dark = Image.new("RGB", region.size, (8, 11, 18))
    canvas.paste(Image.blend(region, dark, opacity), box)


def _draw_camera_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    scenario: DemoScenario,
    step: int,
    config: RenderConfig,
    scale: float,
) -> None:
    left, top, right, bottom = box
    if config.redact_camera:
        draw.rectangle(box, fill="#0B1220")
        hatch_spacing = round(28 * scale)
        for offset in range(0, (right - left) + (bottom - top), hatch_spacing):
            draw.line((left + offset, top, left, top + offset), fill="#16233A", width=1)
            draw.line((right, top + offset - (right - left), left + offset, bottom), fill="#16233A", width=1)
        center = ((left + right) // 2, (top + bottom) // 2)
        draw.text((center[0], center[1] - round(12 * scale)), "Camera feed omitted", font=_font(round(20 * scale), bold=True), fill="#94A3B8", anchor="mm")
        draw.text((center[0], center[1] + round(16 * scale)), "PhysicalAI-AV dataset license forbids public redistribution", font=_font(round(13 * scale)), fill="#64748B", anchor="mm")
    else:
        frame = scenario.camera.frames[min(step, len(scenario.camera.frames) - 1)]
        fitted = ImageOps.fit(frame, (right - left, bottom - top), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (left, top))
    draw.rectangle(box, outline=PANEL_BORDER, width=2)

    pad = round(14 * scale)
    overlay = (left + pad, top + pad, left + round(430 * scale), top + round(158 * scale))
    _overlay_box(canvas, overlay, 0.72)
    draw.rectangle(overlay, outline="#334155", width=1)
    text_x = overlay[0] + round(12 * scale)
    text_y = overlay[1] + round(10 * scale)
    line_h = round(24 * scale)
    draw.text((text_x, text_y), "Alpamayo Trajectory Renderer", font=_font(round(17 * scale), bold=True), fill=WHITE)
    total_steps = len(scenario.ground_truth_xy_m)
    num_samples = len(scenario.sample_xy_m)
    best_ade = float(
        np.linalg.norm(
            scenario.sample_xy_m[scenario.best_index] - scenario.ground_truth_xy_m, axis=1
        ).mean()
    )
    lines = [
        f"Time: {(step + 1) / config.fps:.2f}s  |  Step: {step + 1}/{total_steps}",
        f"Latency: {scenario.stage_latencies_ms.total / 1000:.2f}s  |  minADE: {best_ade:.3f}m",
        f"Trajectory sample: {num_samples}/{num_samples}",
    ]
    for index, line in enumerate(lines):
        draw.text((text_x, text_y + line_h * (index + 1)), line, font=_font(round(14 * scale)), fill="#CBD5E1")
    legend_y = text_y + line_h * (len(lines) + 1)
    swatch = round(16 * scale)
    draw.line((text_x, legend_y + swatch // 2, text_x + swatch, legend_y + swatch // 2), fill=PREDICTED, width=3)
    draw.text((text_x + swatch + round(6 * scale), legend_y), "Predicted", font=_font(round(13 * scale)), fill=WHITE)
    gt_x = text_x + round(150 * scale)
    draw.line((gt_x, legend_y + swatch // 2, gt_x + swatch, legend_y + swatch // 2), fill=GROUND_TRUTH, width=3)
    draw.text((gt_x + swatch + round(6 * scale), legend_y), "Ground Truth", font=_font(round(13 * scale)), fill=WHITE)


def _draw_brand_bar(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], scale: float) -> None:
    _panel(draw, box, radius=round(10 * scale), fill=BRAND_BG)
    left, top, right, bottom = box
    mid_y = (top + bottom) // 2
    pad = round(18 * scale)
    yonsei_font = _font(round(24 * scale), bold=True)
    draw.text((left + pad, mid_y), "YONSEI UNIVERSITY", font=yonsei_font, fill=BRAND_BLUE, anchor="lm")
    yonsei_width = draw.textbbox((0, 0), "YONSEI UNIVERSITY", font=yonsei_font)[2]
    divider_x = left + pad + yonsei_width + round(18 * scale)
    draw.line((divider_x, top + round(10 * scale), divider_x, bottom - round(10 * scale)), fill="#CBD5E1", width=2)
    draw.text(
        (divider_x + round(18 * scale), mid_y),
        "yax",
        font=_font(round(26 * scale), bold=True),
        fill="#0F172A",
        anchor="lm",
    )


def _wrap_to_pixel_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Word-wrap using measured glyph widths instead of a character-count estimate."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _draw_text_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    body: str,
    scale: float,
) -> None:
    _panel(draw, box, radius=round(12 * scale))
    pad_x = round(18 * scale)
    draw.text((box[0] + pad_x, box[1] + round(10 * scale)), label, font=_font(round(14 * scale), bold=True), fill=ON_WHITE_ACCENT)
    body_font = _font(round(17 * scale), bold=True)
    max_width = (box[2] - box[0]) - 2 * pad_x
    wrapped = _wrap_to_pixel_width(draw, body, body_font, max_width)
    draw.multiline_text(
        (box[0] + pad_x, box[1] + round(34 * scale)),
        wrapped,
        font=body_font,
        fill=INK,
        spacing=round(5 * scale),
    )


def _panel_header(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, color: str, scale: float) -> int:
    left, top, right, _ = box
    header_h = round(26 * scale)
    draw.rounded_rectangle((left, top, right, top + header_h), radius=round(8 * scale), fill=color)
    draw.rectangle((left, top + header_h // 2, right, top + header_h), fill=color)
    draw.text((left + round(10 * scale), top + header_h // 2), title, font=_font(round(13 * scale), bold=True), fill=WHITE, anchor="lm")
    return top + header_h + round(8 * scale)


def _draw_resource_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], resources: ResourceSnapshot, scale: float) -> None:
    _panel(draw, box, radius=round(10 * scale))
    content_top = _panel_header(draw, box, "PROCESS RESOURCE SUMMARY", STAGE_COLORS["vision"], scale)
    rows = [
        ("CPU %", resources.cpu_percent, 100.0),
        ("RSS MB", resources.rss_mb, max(resources.rss_mb * 1.2, 1024.0)),
        ("Threads", float(resources.num_threads), max(resources.num_threads * 1.2, 32.0)),
    ]
    pad = round(14 * scale)
    bar_left = box[0] + round(90 * scale)
    bar_right = box[2] - round(70 * scale)
    row_h = round(26 * scale)
    for index, (label, value, scale_max) in enumerate(rows):
        y = content_top + index * row_h
        draw.text((box[0] + pad, y), label, font=_font(round(13 * scale)), fill=MUTED)
        bar_h = round(12 * scale)
        draw.rounded_rectangle((bar_left, y + 2, bar_right, y + 2 + bar_h), radius=bar_h // 2, fill=GRID_LINE)
        fraction = min(value / scale_max, 1.0) if scale_max > 0 else 0.0
        filled_right = bar_left + round((bar_right - bar_left) * fraction)
        if filled_right > bar_left:
            draw.rounded_rectangle((bar_left, y + 2, filled_right, y + 2 + bar_h), radius=bar_h // 2, fill=STAGE_COLORS["vision"])
        draw.text((bar_right + round(8 * scale), y), f"{value:.1f}", font=_font(round(13 * scale), bold=True), fill=INK)


def _draw_cpu_core_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], resources: ResourceSnapshot, scale: float) -> None:
    _panel(draw, box, radius=round(10 * scale))
    content_top = _panel_header(draw, box, "CPU CORE UTILIZATION", STAGE_COLORS["prefill"], scale)
    cores = resources.per_core_percent
    pad = round(14 * scale)
    plot = (box[0] + pad, content_top, box[2] - pad, box[3] - round(22 * scale))
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=GRID_LINE, width=1)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=GRID_LINE, width=1)
    if cores:
        gap = max(round(2 * scale), 1)
        bar_w = max((plot[2] - plot[0] - gap * (len(cores) - 1)) / len(cores), 2.0)
        for index, percent in enumerate(cores):
            bar_h = (plot[3] - plot[1]) * min(percent, 100.0) / 100.0
            x0 = plot[0] + index * (bar_w + gap)
            draw.rectangle((x0, plot[3] - bar_h, x0 + bar_w, plot[3]), fill=STAGE_COLORS["prefill"])
    draw.text((plot[0], plot[3] + round(2 * scale)), "core id", font=_font(round(11 * scale)), fill=MUTED)


def _draw_latency_minade_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    scenario: DemoScenario,
    scale: float,
) -> None:
    _panel(draw, box, radius=round(10 * scale))
    content_top = _panel_header(draw, box, "LATENCY _ MINADE", STAGE_COLORS["total"], scale)
    latencies = scenario.per_sample_latency_ms
    ades = [
        float(np.linalg.norm(sample - scenario.ground_truth_xy_m, axis=1).mean())
        for sample in scenario.sample_xy_m
    ]
    pad = round(14 * scale)
    plot = (box[0] + pad, content_top, box[2] - pad, box[3] - round(34 * scale))
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=GRID_LINE, width=1)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=GRID_LINE, width=1)

    def normalize(values: list[float]) -> list[float]:
        lo, hi = min(values), max(values)
        span = hi - lo if hi > lo else 1.0
        return [(v - lo) / span for v in values]

    def to_points(values: list[float]) -> list[tuple[float, float]]:
        count = max(len(values) - 1, 1)
        return [
            (plot[0] + (index / count) * (plot[2] - plot[0]), plot[3] - value * (plot[3] - plot[1]))
            for index, value in enumerate(values)
        ]

    line_width = max(round(2.4 * scale), 2)
    if len(latencies) > 1:
        draw.line(to_points(normalize(list(latencies))), fill=STAGE_COLORS["decode"], width=line_width, joint="curve")
    if len(ades) > 1:
        draw.line(to_points(normalize(ades)), fill=GROUND_TRUTH, width=line_width, joint="curve")

    label_font = _font(round(12 * scale), bold=True)
    draw.text((plot[0], plot[3] + round(6 * scale)), f"Latency {latencies[-1]:.0f}ms", font=label_font, fill=STAGE_COLORS["decode"])
    draw.text((plot[0] + round(130 * scale), plot[3] + round(6 * scale)), f"minADE {ades[-1]:.2f}m", font=label_font, fill=GROUND_TRUTH)
    draw.text((plot[0], plot[1] - round(2 * scale)), "sample id →", font=_font(round(10 * scale)), fill=MUTED, anchor="ls")


def _draw_latency_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], latencies: StageLatenciesMs, scale: float) -> None:
    _panel(draw, box, radius=round(10 * scale))
    content_top = _panel_header(draw, box, "LATENCY BY STAGE", STAGE_COLORS["decode"], scale)
    rows = [
        ("Vision", latencies.vision, STAGE_COLORS["vision"]),
        ("Prefill", latencies.prefill, STAGE_COLORS["prefill"]),
        ("Decode", latencies.decode, STAGE_COLORS["decode"]),
        ("Diffusion", latencies.diffusion, STAGE_COLORS["diffusion"]),
        ("Total", latencies.total, STAGE_COLORS["total"]),
    ]
    pad = round(14 * scale)
    bar_left = box[0] + round(72 * scale)
    bar_right = box[2] - round(80 * scale)
    row_h = round(20 * scale)
    max_ms = max(latencies.total, 1.0)
    for index, (label, value, color) in enumerate(rows):
        y = content_top + index * row_h
        draw.text((box[0] + pad, y), label, font=_font(round(12 * scale)), fill=MUTED)
        bar_h = round(10 * scale)
        fraction = min(value / max_ms, 1.0)
        filled_right = bar_left + round((bar_right - bar_left) * fraction)
        draw.rounded_rectangle((bar_left, y + 2, bar_right, y + 2 + bar_h), radius=bar_h // 2, fill=GRID_LINE)
        if filled_right > bar_left:
            draw.rounded_rectangle((bar_left, y + 2, filled_right, y + 2 + bar_h), radius=bar_h // 2, fill=color)
        draw.text((bar_right + round(8 * scale), y), f"{value:.0f} ms", font=_font(round(12 * scale), bold=True), fill=INK)


def render_frame(scenario: DemoScenario, config: RenderConfig, step: int) -> Image.Image:
    """Render one synchronized camera-and-trajectory dashboard frame."""
    canvas = _gradient_background(config)
    draw = ImageDraw.Draw(canvas)
    scale = config.width / 1920
    margin = round(24 * scale)
    gap = round(14 * scale)

    row1_bottom = round(config.height * 0.655)
    row2_top = row1_bottom + gap
    row2_bottom = row2_top + round(config.height * 0.095)
    row3_top = row2_bottom + gap
    row3_bottom = config.height - margin

    camera_right = round(config.width * 0.63)
    plot_left = camera_right + gap

    camera_box = (margin, margin, camera_right, row1_bottom)
    _draw_camera_panel(canvas, draw, camera_box, scenario, step, config, scale)

    brand_h = round(config.height * 0.052)
    brand_box = (plot_left, margin, config.width - margin, margin + brand_h)
    _draw_brand_bar(draw, brand_box, scale)

    plots_top = brand_box[3] + gap
    _section_label(draw, (plot_left, plots_top), "- Best Trajectory", scale)
    _section_label(draw, (plot_left + (config.width - margin - plot_left) // 2 + gap // 2, plots_top), "- Overall Trajectory", scale)
    plot_row = (plots_top + round(26 * scale), row1_bottom)
    plot_width = (config.width - margin - plot_left - gap) // 2
    best_plot_box = (plot_left, plot_row[0], plot_left + plot_width, plot_row[1])
    overall_plot_box = (plot_left + plot_width + gap, plot_row[0], config.width - margin, plot_row[1])
    _panel(draw, best_plot_box, radius=round(10 * scale))
    _panel(draw, overall_plot_box, radius=round(10 * scale))

    bounds = _xy_bounds(scenario.history_xy_m, scenario.ground_truth_xy_m, *scenario.sample_xy_m)
    best_xy = scenario.sample_xy_m[scenario.best_index]
    _draw_traj_plot(draw, best_plot_box, "BEST", bounds, scenario.history_xy_m, scenario.ground_truth_xy_m, (), best_xy, step, scale)
    others = tuple(s for i, s in enumerate(scenario.sample_xy_m) if i != scenario.best_index)
    _draw_traj_plot(draw, overall_plot_box, "OVERALL", bounds, scenario.history_xy_m, scenario.ground_truth_xy_m, others, best_xy, step, scale)

    reasoning_box = (margin, row2_top, camera_right, row2_bottom)
    _draw_text_panel(draw, reasoning_box, "- REASONING (CHAIN OF CAUSATION)", scenario.reasoning, scale)
    nav_box = (plot_left, row2_top, config.width - margin, row2_bottom)
    _draw_text_panel(draw, nav_box, "- NAVIGATION SUMMARY", _navigation_summary(best_xy), scale)

    panel_width = (config.width - 2 * margin - 3 * gap) // 4
    panel_boxes = [
        (margin + index * (panel_width + gap), row3_top, margin + index * (panel_width + gap) + panel_width, row3_bottom)
        for index in range(4)
    ]
    _draw_resource_panel(draw, panel_boxes[0], scenario.resources, scale)
    _draw_cpu_core_panel(draw, panel_boxes[1], scenario.resources, scale)
    _draw_latency_minade_panel(draw, panel_boxes[2], scenario, scale)
    _draw_latency_panel(draw, panel_boxes[3], scenario.stage_latencies_ms, scale)

    return canvas


def encode_demo_video(
    scenario: DemoScenario,
    output_path: Path,
    config: RenderConfig = RenderConfig(),
) -> None:
    """Render all trajectory steps and encode them as H.264 MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alpamayo-demo-") as temporary_directory:
        frame_pattern = Path(temporary_directory) / "frame_%04d.png"
        for step in range(len(scenario.ground_truth_xy_m)):
            render_frame(scenario, config, step).save(Path(temporary_directory) / f"frame_{step:04d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(config.fps),
                "-i", str(frame_pattern), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
            ],
            check=True,
        )


def _stage_timer_hooks(model: object, stage_ms: dict[str, float]) -> list[object]:
    """Install forward hooks that measure wall-clock time inside each named submodule."""
    import torch

    starts: dict[int, float] = {}

    def make_pre(_bucket: str):
        def hook(module: object, args: tuple, kwargs: dict) -> None:
            torch.cuda.synchronize()
            starts[id(module)] = time.perf_counter()

        return hook

    def make_post(bucket: str):
        def hook(module: object, args: tuple, kwargs: dict, output: object) -> None:
            torch.cuda.synchronize()
            stage_ms[bucket] += (time.perf_counter() - starts[id(module)]) * 1000.0

        return hook

    def post_language_model(module: object, args: tuple, kwargs: dict, output: object) -> None:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - starts[id(module)]) * 1000.0
        seq_len = None
        if kwargs.get("input_ids") is not None:
            seq_len = kwargs["input_ids"].shape[1]
        elif kwargs.get("inputs_embeds") is not None:
            seq_len = kwargs["inputs_embeds"].shape[1]
        elif args:
            seq_len = args[0].shape[1]
        bucket = "prefill" if (seq_len or 0) > 1 else "decode"
        stage_ms[bucket] += elapsed_ms

    vision = model.vlm.model.visual
    language_model = model.vlm.model.language_model
    expert = model.expert
    return [
        vision.register_forward_pre_hook(make_pre("vision"), with_kwargs=True),
        vision.register_forward_hook(make_post("vision"), with_kwargs=True),
        language_model.register_forward_pre_hook(make_pre("lm"), with_kwargs=True),
        language_model.register_forward_hook(post_language_model, with_kwargs=True),
        expert.register_forward_pre_hook(make_pre("diffusion"), with_kwargs=True),
        expert.register_forward_hook(make_post("diffusion"), with_kwargs=True),
    ]


def _monitor_resources(stop: threading.Event, samples: list[tuple[float, float, int]]) -> None:
    import psutil

    process = psutil.Process()
    process.cpu_percent()
    while not stop.is_set():
        cpu = process.cpu_percent(interval=0.1)
        samples.append((cpu, process.memory_info().rss / (1024 ** 2), process.num_threads()))


def generate_demo_scenario(
    clip_id: str = DEFAULT_CLIP_ID,
    num_traj_samples: int = DEFAULT_NUM_SAMPLES,
) -> DemoScenario:
    """Run Alpamayo inference and load synchronized future camera frames."""
    import physical_ai_av
    import psutil
    import torch

    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000, avdi=avdi)
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        helper.create_message(data["image_frames"].flatten(0, 1)),
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    stage_ms = {"vision": 0.0, "prefill": 0.0, "decode": 0.0, "diffusion": 0.0}
    hooks = _stage_timer_hooks(model, stage_ms)
    resource_samples: list[tuple[float, float, int]] = []
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=_monitor_resources, args=(stop_monitor, resource_samples), daemon=True)
    monitor_thread.start()

    # The 10B-parameter model already occupies ~22GB in bf16, leaving too little headroom
    # for a batched num_traj_samples>1 call on a 24GB GPU (pixel_values are duplicated per
    # sample and OOM the vision tower). Instead, sample one rollout at a time and accumulate.
    sample_xy_list: list[FloatArray] = []
    cot_list: list[str] = []
    per_sample_latency_ms: list[float] = []
    for rollout_index in range(num_traj_samples):
        rollout_start = time.perf_counter()
        torch.cuda.manual_seed_all(42 + rollout_index)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=copy.deepcopy(model_inputs), top_p=0.98, temperature=0.6,
                num_traj_samples=1, max_generation_length=256, return_extra=True,
            )
        torch.cuda.synchronize()
        per_sample_latency_ms.append((time.perf_counter() - rollout_start) * 1000.0)
        sample_xy_list.append(pred_xyz.detach().cpu().numpy()[0, 0, 0, :, :2].astype(np.float32))
        cot_list.extend(_flatten_strings(extra["cot"][0]))
        del pred_xyz, _pred_rot, extra
        torch.cuda.empty_cache()

    stop_monitor.set()
    monitor_thread.join(timeout=1.0)
    for handle in hooks:
        handle.remove()

    per_core = tuple(float(value) for value in psutil.cpu_percent(percpu=True))
    if resource_samples:
        avg_cpu = sum(sample[0] for sample in resource_samples) / len(resource_samples)
        avg_rss = sum(sample[1] for sample in resource_samples) / len(resource_samples)
        threads = resource_samples[-1][2]
    else:
        process = psutil.Process()
        avg_cpu, avg_rss, threads = process.cpu_percent(), process.memory_info().rss / (1024 ** 2), process.num_threads()
    resources = ResourceSnapshot(cpu_percent=avg_cpu, rss_mb=avg_rss, num_threads=threads, per_core_percent=per_core)
    latencies = StageLatenciesMs(**stage_ms)

    sample_xy = np.stack(sample_xy_list, axis=0)
    ground_truth_xy = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2].astype(np.float32)
    history_xy = data["ego_history_xyz"].cpu().numpy()[0, 0, :, :2].astype(np.float32)
    per_sample_ade = np.linalg.norm(sample_xy - ground_truth_xy[None], axis=-1).mean(axis=-1)
    best_index = int(np.argmin(per_sample_ade))
    cot_strings = cot_list

    timestamps = 5_100_000 + np.arange(1, sample_xy.shape[1] + 1, dtype=np.int64) * 100_000
    camera = avdi.get_clip_feature(clip_id, avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV, maybe_stream=True)
    frames, _frame_timestamps = camera.decode_images_from_timestamps(timestamps)
    resized = tuple(
        ImageOps.fit(Image.fromarray(frame).convert("RGB"), (1280, 720), method=Image.Resampling.LANCZOS)
        for frame in frames
    )

    return DemoScenario(
        clip_id=clip_id,
        reasoning=cot_strings[best_index] if best_index < len(cot_strings) else cot_strings[0],
        camera=CameraVideo(name="FRONT WIDE • 120°", frames=resized),
        num_cameras=4,
        backbone_summary="Qwen3-VL-8B backbone + 2.3B flow-matching expert",
        history_xy_m=history_xy,
        ground_truth_xy_m=ground_truth_xy,
        sample_xy_m=tuple(sample_xy[index] for index in range(sample_xy.shape[0])),
        best_index=best_index,
        stage_latencies_ms=latencies,
        per_sample_latency_ms=tuple(per_sample_latency_ms),
        resources=resources,
    )


def main() -> None:
    """Generate the repository's default Alpamayo demo video."""
    print(f"Running Alpamayo inference for {DEFAULT_CLIP_ID}...")
    scenario = generate_demo_scenario()
    print(f"Rendering {len(scenario.ground_truth_xy_m)} synchronized frames...")
    encode_demo_video(scenario, DEFAULT_OUTPUT)
    print(f"Saved demo video: {DEFAULT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
