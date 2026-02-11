#!/usr/bin/env python3
"""Run person-only instance segmentation on Raspberry Pi camera using Hailo-10H.

Defaults:
  - input source: Raspberry Pi camera (CSI)
  - architecture: hailo10h

Display is handled by the normal GStreamer sink (autovideosink by default), so
you get an on-device output window/sink rather than an OpenCV callback window.
"""

# region imports
import argparse

import gi
import hailo

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from hailo_apps.hailo_app_python.apps.instance_segmentation.instance_segmentation_pipeline import (
    GStreamerInstanceSegmentationApp,
)
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class

# endregion imports


hailo_logger = get_logger(__name__)


class PersonSegmentationCallbackData(app_callback_class):
    """Container for user callback configuration and state."""

    def __init__(self, confidence_threshold: float):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.mask_only = False
        self._reported = False


class PersonDisplayInstanceSegmentationApp(GStreamerInstanceSegmentationApp):
    """Standard segmentation app with configurable display sink.

    This keeps the built-in hailooverlay display path, which is more robust for
    Raspberry Pi on-device display than OpenCV callback rendering.
    """

    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser=parser)

        if getattr(self.options_menu, "video_sink", None):
            self.video_sink = self.options_menu.video_sink
            self.create_pipeline()

    def get_pipeline_string(self):
        pipeline_string = super().get_pipeline_string()

        # In --mask-only mode, darken the frame before hailooverlay draws masks.
        # This keeps rendering in GStreamer (no OpenCV window/Qt dependency).
        if getattr(self.options_menu, "mask_only", False):
            parts = pipeline_string.split(" ! ")
            overlay_index = None
            for i, part in enumerate(parts):
                if "hailooverlay name=hailo_display_overlay" in part:
                    overlay_index = i
                    break

            if overlay_index is not None:
                parts.insert(
                    overlay_index,
                    "videobalance name=mask_only_balance brightness=-1.0 contrast=0.0 saturation=0.0",
                )
                pipeline_string = " ! ".join(parts)
            elif not self.user_data._reported:
                hailo_logger.warning(
                    "Could not locate hailo_display_overlay in pipeline; --mask-only won't darken feed"
                )
                self.user_data._reported = True

        return pipeline_string


def app_callback(pad, info, user_data: PersonSegmentationCallbackData):
    """Filter metadata so overlay displays only person detections."""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    roi = hailo.get_roi_from_buffer(buffer)
    detections = list(roi.get_objects_typed(hailo.HAILO_DETECTION))

    person_count = 0

    for detection in detections:
        if (
            detection.get_label() != "person"
            or detection.get_confidence() < user_data.confidence_threshold
        ):
            try:
                roi.remove_object(detection)
            except Exception as err:  # noqa: BLE001
                if not user_data._reported:
                    hailo_logger.warning(
                        "Could not remove non-person detection from ROI metadata: %s",
                        err,
                    )
                    user_data._reported = True
            continue
        person_count += 1

    if person_count and user_data.get_count() % 30 == 0:
        hailo_logger.info("Persons in frame: %d", person_count)

    return Gst.PadProbeReturn.OK


def build_parser() -> argparse.ArgumentParser:
    parser = get_default_parser()
    parser.description = "Person-only segmentation from RPi camera on Hailo-10H"
    parser.set_defaults(input="rpi", arch="hailo10h", use_frame=False, frame_rate=10)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum confidence for person detections (default: 0.4)",
    )
    parser.add_argument(
        "--video-sink",
        type=str,
        default="autovideosink",
        help="GStreamer video sink for display (e.g. autovideosink, ximagesink, waylandsink, kmssink)",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Display person segmentation masks only (black background, no camera video feed)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    confidence_threshold = min(1.0, max(0.0, args.confidence_threshold))

    user_data = PersonSegmentationCallbackData(
        confidence_threshold=confidence_threshold,
    )
    user_data.mask_only = args.mask_only

    app = PersonDisplayInstanceSegmentationApp(app_callback, user_data, parser=parser)
    app.run()


if __name__ == "__main__":
    main()
