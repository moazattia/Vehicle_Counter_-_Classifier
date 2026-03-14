"""Core CV pipeline: detect → track → count → annotate → report."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO

import config

# ── Shared model (loaded once at startup) ──────────────
_model: YOLO | None = None


def load_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(config.MODEL_SIZE)
    return _model


def get_device() -> str:
    model = load_model()
    return str(model.device)


def _write_status(status_file: Path, data: dict):
    """Atomically write status JSON (write tmp then rename)."""
    tmp = status_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(status_file)


# ── Pipeline ───────────────────────────────────────────

def run_pipeline(
    input_video: Path,
    output_dir: Path,
    job_id: str,
    line_position: float = config.LINE_POSITION,
    confidence: float = config.CONFIDENCE_THRESHOLD,
    progress_cb=None,
) -> dict:
    """Run full detection + tracking + counting pipeline.

    Returns a dict with counts, duration, and output file paths.
    """
    model = load_model()
    status_file = output_dir / "status.json"

    # ── Video info ─────────────────────────────────────
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total_frames / fps if fps else 0.0

    # ── Output paths ──────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out = output_dir / f"{job_id}_annotated.mp4"
    csv_out = output_dir / f"{job_id}_events.csv"
    report_out = output_dir / f"{job_id}_report.md"
    chart_out = output_dir / f"{job_id}_chart.png"

    # ── Counting line ─────────────────────────────────
    line_y = int(h * line_position)
    line_start = sv.Point(x=0, y=line_y)
    line_end = sv.Point(x=w, y=line_y)
    line_zone = sv.LineZone(start=line_start, end=line_end)

    # ── Tracker & annotators ──────────────────────────
    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.5)

    # ── CSV writer ────────────────────────────────────
    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp_sec", "vehicle_type", "tracker_id"])

    # ── Counting state ────────────────────────────────
    counts: dict[str, int] = {name: 0 for name in config.VEHICLE_CLASSES.values()}
    crossed_ids: set[int] = set()
    events: list[tuple[float, str, int]] = []

    # ── Video writer ──────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (w, h))

    # ── Frame loop ────────────────────────────────────
    cap = cv2.VideoCapture(str(input_video))
    frame_idx = 0
    proc_start = time.time()
    frame_skip = getattr(config, "FRAME_SKIP", 2)
    last_detections = sv.Detections.empty()
    last_labels: list[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        run_detection = (frame_idx % frame_skip == 0)

        if run_detection:
            # detect
            results = model(frame, conf=confidence, verbose=False)[0]

            # filter to vehicle classes only
            mask = np.isin(results.boxes.cls.cpu().numpy(), list(config.VEHICLE_CLASSES.keys()))
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[mask]

            # track
            detections = tracker.update_with_detections(detections)

            # count crossings
            line_zone.trigger(detections)

            if detections.tracker_id is not None:
                for i, tid in enumerate(detections.tracker_id):
                    if tid in crossed_ids:
                        continue
                    bbox = detections.xyxy[i]
                    cy = (bbox[1] + bbox[3]) / 2
                    if abs(cy - line_y) < h * 0.02:
                        cls_id = int(detections.class_id[i])
                        vtype = config.VEHICLE_CLASSES.get(cls_id, "Unknown")
                        counts[vtype] = counts.get(vtype, 0) + 1
                        crossed_ids.add(tid)
                        ts = frame_idx / fps
                        events.append((ts, vtype, int(tid)))
                        csv_writer.writerow([f"{ts:.2f}", vtype, tid])

            # build labels
            labels = []
            if detections.tracker_id is not None:
                for tid, cls_id, conf_score in zip(
                    detections.tracker_id, detections.class_id, detections.confidence
                ):
                    vtype = config.VEHICLE_CLASSES.get(int(cls_id), "?")
                    labels.append(f"#{tid} {vtype} {conf_score:.2f}")

            last_detections = detections
            last_labels = labels
        else:
            detections = last_detections
            labels = last_labels

        # ── Annotate frame ────────────────────────────
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)

        # ── Live counter panel ────────────────────────
        elapsed = time.time() - proc_start
        current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
        panel_lines = [
            f"FPS: {current_fps:.1f}",
            f"Total: {sum(counts.values())}",
        ] + [f"{k}: {v}" for k, v in counts.items()]

        y_offset = 30
        for line in panel_lines:
            cv2.putText(annotated, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            y_offset += 28

        writer.write(annotated)
        frame_idx += 1

        # progress callback
        if progress_cb and total_frames > 0:
            progress_cb(frame_idx / total_frames * 100)

        # write progress to status file every 30 frames
        if frame_idx % 30 == 0 and total_frames > 0:
            _write_status(status_file, {
                "status": "processing",
                "progress_percent": round(frame_idx / total_frames * 100, 1),
                "counts": counts,
            })

    cap.release()
    writer.release()
    csv_file.close()

    # ── Generate report ───────────────────────────────
    total = sum(counts.values())
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report = f"""# Vehicle Counting Report

**Job ID:** {job_id}
**Processed at:** {now_str}
**Video duration:** {duration:.1f}s | **Total frames:** {total_frames} | **FPS:** {fps:.1f}

## Counts

| Vehicle Type | Count |
|---|---|
"""
    for vtype, cnt in counts.items():
        report += f"| {vtype} | {cnt} |\n"
    report += f"| **Total** | **{total}** |\n"

    report += f"\n## Configuration\n- Line position: {line_position}\n- Confidence: {confidence}\n"

    report_out.write_text(report, encoding="utf-8")

    # ── Generate chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    types = list(counts.keys())
    vals = list(counts.values())
    colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c"]
    bars = ax.bar(types, vals, color=colors[: len(types)])
    ax.set_title("Vehicle Counts by Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(chart_out), dpi=150)
    plt.close(fig)

    result = {
        "counts": counts,
        "total": total,
        "duration": duration,
        "processed_at": now_str,
        "video_out": str(video_out),
        "csv_out": str(csv_out),
        "report_out": str(report_out),
        "chart_out": str(chart_out),
    }

    # write final "done" status
    _write_status(status_file, {
        "status": "done",
        "progress_percent": 100.0,
        "counts": counts,
        "total": total,
        "duration": duration,
        "processed_at": now_str,
        "video_out": str(video_out),
        "csv_out": str(csv_out),
        "report_out": str(report_out),
        "chart_out": str(chart_out),
    })

    return result
