#!/usr/bin/env python3
"""
remote_glass.py (hailo-apps v25.10.0-oriented)

Raspberry Pi 5 + AI HAT+ 2 (Hailo) + Pi Camera:
- Runs instance segmentation on a live stream
- Keeps ONLY "person"
- Draws ONLY the outline (contours) of each person mask (no filled highlight)

Run (from hailo-apps repo, after sourcing env + venv):
  source setup.env.sh
  python remote_glass.py --input rpi

Notes:
- This targets the newer hailo-apps framework layout (v25.x) using hailo_apps.* imports.
- If your local class/module paths differ slightly, the import block below tries a few common options.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import cv2

import gi

import sys

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import hailo  # noqa: E402


# -----------------------------------------------------------------------------
# hailo-apps v25.x imports (try a few paths; repo layout can differ slightly)
# -----------------------------------------------------------------------------
def _import_hailo_apps_symbols():
    # Buffer utils
    get_caps_from_pad = None
    get_numpy_from_buffer = None
    app_callback_class = None
    GStreamerInstanceSegmentationApp = None

    # ---- buffer utils + callback base ----
    candidate_common = [
        # As suggested for newer framework layouts
        (
            "hailo_apps.hailo_app_python.core.common.buffer_utils",
            "get_caps_from_pad",
            "get_numpy_from_buffer",
        ),
        # Alternate names seen in some layouts
        (
            "hailo_apps.hailo_app_python.core.common.gstreamer_utils",
            "get_caps_from_pad",
            "get_numpy_from_buffer",
        ),
    ]

    candidate_callback = [
        ("hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app", "app_callback_class"),
        ("hailo_apps.hailo_app_python.core.gstreamer.app", "app_callback_class"),
    ]

    candidate_pipelines = [
        # “apps” style
        (
            "hailo_apps.hailo_app_python.apps.instance_segmentation.instance_segmentation_pipeline",
            "GStreamerInstanceSegmentationApp",
        ),
        (
            "hailo_apps.hailo_app_python.apps.instance_segmentation.pipeline",
            "GStreamerInstanceSegmentationApp",
        ),
        # “python/pipeline_apps” style (entrypoint style)
        (
            "hailo_apps.python.pipeline_apps.instance_segmentation.instance_segmentation",
            "GStreamerInstanceSegmentationApp",
        ),
        (
            "hailo_apps.python.pipeline_apps.instance_segmentation.instance_segmentation_pipeline",
            "GStreamerInstanceSegmentationApp",
        ),
    ]

    import importlib

    # common utils
    for mod, a, b in candidate_common:
        try:
            m = importlib.import_module(mod)
            get_caps_from_pad = getattr(m, a)
            get_numpy_from_buffer = getattr(m, b)
            break
        except Exception:
            pass

    # callback base class
    for mod, a in candidate_callback:
        try:
            m = importlib.import_module(mod)
            app_callback_class = getattr(m, a)
            break
        except Exception:
            pass

    # pipeline app
    for mod, cls in candidate_pipelines:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, cls):
                GStreamerInstanceSegmentationApp = getattr(m, cls)
                break
        except Exception:
            pass

    missing = []
    if get_caps_from_pad is None or get_numpy_from_buffer is None:
        missing.append("buffer utils (get_caps_from_pad/get_numpy_from_buffer)")
    if app_callback_class is None:
        missing.append("app_callback_class")
    if GStreamerInstanceSegmentationApp is None:
        missing.append("GStreamerInstanceSegmentationApp")

    if missing:
        raise ModuleNotFoundError(
            "Could not import required hailo-apps v25.x symbols: "
            + ", ".join(missing)
            + "\n\n"
            "This usually means your local v25.10.0 tree uses slightly different module paths.\n"
            "Quick way to locate the right paths:\n"
            "  python -c \"import pkgutil, hailo_apps; "
            "print([m.name for m in pkgutil.walk_packages(hailo_apps.__path__, hailo_apps.__name__+'.') "
            "if 'instance_seg' in m.name])\"\n"
            "  python -c \"import hailo_apps, pkgutil; "
            "print([m.name for m in pkgutil.walk_packages(hailo_apps.__path__, hailo_apps.__name__+'.') "
            "if 'buffer_utils' in m.name or 'gstreamer_app' in m.name])\""
        )

    return get_caps_from_pad, get_numpy_from_buffer, app_callback_class, GStreamerInstanceSegmentationApp


get_caps_from_pad, get_numpy_from_buffer, app_callback_class, GStreamerInstanceSegmentationApp = _import_hailo_apps_symbols()


# -----------------------------------------------------------------------------
# User callback data class (inherits the helper that manages display, counters, etc.)
# -----------------------------------------------------------------------------
class UserData(app_callback_class):
    def __init__(self):
        super().__init__()


# -----------------------------------------------------------------------------
# Mask/contour helpers
# -----------------------------------------------------------------------------
def _parse_bgr(s: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError('outline-color must be "B,G,R" (e.g. 0,255,0)')
    b, g, r = (int(parts[0]), int(parts[1]), int(parts[2]))
    b = max(0, min(255, b))
    g = max(0, min(255, g))
    r = max(0, min(255, r))
    return (b, g, r)


def _clamp_bbox(bbox, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x0 = int(bbox.xmin() * frame_w)
    y0 = int(bbox.ymin() * frame_h)
    x1 = int(bbox.xmax() * frame_w)
    y1 = int(bbox.ymax() * frame_h)

    x0 = max(0, min(frame_w - 1, x0))
    y0 = max(0, min(frame_h - 1, y0))
    x1 = max(0, min(frame_w, x1))
    y1 = max(0, min(frame_h, y1))

    if x1 <= x0:
        x1 = min(frame_w, x0 + 1)
    if y1 <= y0:
        y1 = min(frame_h, y0 + 1)

    return x0, y0, x1, y1


def _mask_to_contours(binary_mask: np.ndarray):
    mask_u8 = (binary_mask.astype(np.uint8) * 255)
    contours, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# -----------------------------------------------------------------------------
# GStreamer probe callback factory
# -----------------------------------------------------------------------------
def make_app_callback(
    score_thresh: float,
    mask_thresh: float,
    outline_thickness: int,
    outline_bgr: Tuple[int, int, int],
    smooth: int,
):
    """
    Returns an app_callback(pad, info, user_data) closure with chosen parameters.

    Hailo ROI objects:
      - detections: hailo.HAILO_DETECTION
      - per detection, mask objects: hailo.HAILO_CONF_CLASS_MASK
    """

    def app_callback(pad, info, user_data: UserData):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        user_data.increment()

        fmt, width, height = get_caps_from_pad(pad)

        frame = None
        if user_data.use_frame and fmt and width and height:
            # Usually returns RGB numpy array (Hailo helper)
            frame = get_numpy_from_buffer(buffer, fmt, width, height)

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        if frame is not None:
            # Draw on RGB, then set frame for display
            for det in detections:
                if det.get_label() != "person":
                    continue
                if det.get_confidence() < score_thresh:
                    continue

                bbox = det.get_bbox()
                x0, y0, x1, y1 = _clamp_bbox(bbox, width, height)
                bw = x1 - x0
                bh = y1 - y0
                if bw < 2 or bh < 2:
                    continue

                masks = det.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
                if not masks:
                    continue

                m = masks[0]
                mh = int(m.get_height())
                mw = int(m.get_width())
                if mh <= 0 or mw <= 0:
                    continue

                # mask data -> (mh, mw)
                mask_data = np.array(m.get_data(), dtype=np.float32).reshape((mh, mw))

                # resize mask to bbox size
                mask_resized = cv2.resize(mask_data, (bw, bh), interpolation=cv2.INTER_CUBIC)

                if smooth > 0:
                    k = smooth * 2 + 1
                    mask_resized = cv2.GaussianBlur(mask_resized, (k, k), 0)

                binary = mask_resized > mask_thresh
                contours = _mask_to_contours(binary)
                if not contours:
                    continue

                # Draw outlines only, offset into full image coords
                for c in contours:
                    c = c + np.array([[[x0, y0]]], dtype=c.dtype)
                    # frame is RGB, OpenCV expects BGR; invert tuple for correct appearance
                    cv2.polylines(
                        frame,
                        [c],
                        isClosed=True,
                        color=outline_bgr[::-1],
                        thickness=outline_thickness,
                    )

            # Convert RGB->BGR for display window (common in these apps)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame_bgr)

        return Gst.PadProbeReturn.OK

    return app_callback


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hailo person segmentation (outlines only) for RPi5 + AI HAT+ 2")

    p.add_argument(
        "--input",
        default="rpi",
        help='Input source. Common values: "rpi" (Pi Camera), "usb", "/dev/video0", or a video file path.',
    )

    # Optional overrides (only applied if the pipeline object exposes compatible attributes)
    p.add_argument("--hef-path", default=None, help="Override HEF path (optional).")
    p.add_argument("--labels-json", default=None, help="Override labels JSON path (optional).")

    # Segmentation filtering / drawing
    p.add_argument("--score-thresh", type=float, default=0.35, help="Min confidence for person detections.")
    p.add_argument("--mask-thresh", type=float, default=0.50, help="Threshold for mask binarization.")
    p.add_argument("--outline-thickness", type=int, default=2, help="Outline thickness in pixels.")
    p.add_argument("--outline-color", default="0,255,0", help='Outline color as "B,G,R" (default green).')
    p.add_argument("--smooth", type=int, default=1, help="Mask smoothing radius (0=off).")

    return p


def main() -> int:
    Gst.init(None)

    args = build_argparser().parse_args()
    outline_bgr = _parse_bgr(args.outline_color)

    user_data = UserData()
    cb = make_app_callback(
        score_thresh=args.score_thresh,
        mask_thresh=args.mask_thresh,
        outline_thickness=max(1, args.outline_thickness),
        outline_bgr=outline_bgr,
        smooth=max(0, args.smooth),
    )

    # hailo-apps pipeline apps parse sys.argv internally during app construction.
    # If the user didn't specify input, inject our default/parsed value.
    if not any(a in ("--input", "-i") or a.startswith("--input=") for a in sys.argv):
        sys.argv += ["--input", args.input]


    # Construct the instance-segmentation app from hailo-apps
    app = GStreamerInstanceSegmentationApp(cb, user_data)

    # Best-effort: configure input/model overrides if your pipeline exposes these
    if args.hef_path:
        for attr in ("hef_path", "hef", "network_hef", "model_hef"):
            if hasattr(app, attr):
                try:
                    setattr(app, attr, args.hef_path)
                    break
                except Exception:
                    pass

    if args.labels_json:
        for attr in ("labels_json", "labels", "labels_path"):
            if hasattr(app, attr):
                try:
                    setattr(app, attr, args.labels_json)
                    break
                except Exception:
                    pass

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
