from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from airtime_bias.io.artifact_history import (
    artifact_label,
    list_artifacts,
    load_manifest,
    parameter_fingerprint,
    preferred_artifact_index,
    write_manifest,
)
from airtime_bias.io.loaders import load_config, load_table
from airtime_bias.io.metadata import add_video_paths, load_active_cast, load_episode_parts
from airtime_bias.io.paths import INTERIM_DIR, ensure_project_dirs
from airtime_bias.io.writers import save_table
from airtime_bias.lower_third.ocr import (
    EasyOCRConfig,
    EasyOCRReader,
    annotate_events_with_ocr,
    easyocr_available,
)
from airtime_bias.lower_third.temporal_events import (
    group_lower_third_events,
    scan_lower_thirds_in_video,
)
from airtime_bias.lower_third.visual_detection import LowerThirdROI, LowerThirdVisualConfig
from airtime_bias.video.clip_utils import get_video_duration_seconds
from airtime_bias.visualization.lower_third_plots import (
    plot_lower_third_event_durations,
    plot_lower_third_score_distribution,
    plot_lower_third_timeline,
)

ensure_project_dirs()
st.set_page_config(page_title="Lower-Third Detection", page_icon="🏷️", layout="wide")
st.title("🏷️ Participant Lower-Third Detection")
st.markdown(
    """
Scan the lower-left region independently of scene segmentation. Consecutive visual detections
are grouped into temporal events, and optional OCR can match imperfect text against the active
cast. This route can recover very short commentary moments that a coarse scene table may merge.
"""
)

config = load_config()
defaults = config.get("lower_third_detection", {})
ROOT_DIR = INTERIM_DIR / "lower_thirds"
FRAME_RESULT_DIR = ROOT_DIR / "frame_detections"
EVENT_DIR = ROOT_DIR / "events"
IMAGE_DIR = ROOT_DIR / "images"

parts = add_video_paths(load_episode_parts(), prefer_proxy=True)
active_cast = load_active_cast()
if parts.empty:
    st.warning("Episode-part metadata is unavailable. Complete Episode Setup first.")
    st.stop()
if not parts["selected_video_exists"].all():
    st.error("Some episode video files could not be found.")
    st.dataframe(
        parts.loc[~parts["selected_video_exists"], ["episode_id", "part_id", "selected_video_path"]],
        width="stretch",
    )
    st.stop()

st.subheader("Scan scope")
scope_col_1, scope_col_2, scope_col_3 = st.columns(3)
with scope_col_1:
    episode_id = st.selectbox("Episode", sorted(parts["episode_id"].astype(str).unique()))
with scope_col_2:
    scope_mode = st.radio("Scope", ["Single part", "All episode parts"], horizontal=True)

selected_all_parts = (
    parts[parts["episode_id"].astype(str) == str(episode_id)].copy().sort_values("part_order")
)
selected_part_id: str | None = None
with scope_col_3:
    if scope_mode == "Single part":
        selected_part_id = st.selectbox(
            "Episode part", selected_all_parts["part_id"].astype(str).tolist()
        )
selected_parts = selected_all_parts.copy()
if selected_part_id is not None:
    selected_parts = selected_parts[selected_parts["part_id"].astype(str) == selected_part_id].copy()

st.subheader("Temporal sampling")
temporal_col_1, temporal_col_2, temporal_col_3, temporal_col_4 = st.columns(4)
with temporal_col_1:
    sample_interval = st.slider(
        "Sampling interval (seconds)",
        min_value=0.25,
        max_value=2.00,
        value=float(defaults.get("sample_interval_seconds", 0.50)),
        step=0.25,
        help="0.5 s is a practical baseline; 0.25 s increases recall for very short labels.",
    )
with temporal_col_2:
    development_mode = st.toggle("Development window", value=True)
with temporal_col_3:
    start_offset = st.number_input(
        "Start offset (seconds)", min_value=0.0, value=0.0, step=30.0, disabled=not development_mode
    )
with temporal_col_4:
    development_minutes = st.number_input(
        "Window duration (minutes)", min_value=0.5, max_value=60.0, value=5.0, step=0.5,
        disabled=not development_mode,
    )

st.subheader("Visual detector")
roi_defaults = defaults.get("roi", {})
roi_col_1, roi_col_2, roi_col_3, roi_col_4 = st.columns(4)
with roi_col_1:
    roi_xmin = st.slider("ROI x minimum", 0.0, 0.40, float(roi_defaults.get("xmin", 0.02)), 0.01)
with roi_col_2:
    roi_xmax = st.slider("ROI x maximum", 0.40, 1.0, float(roi_defaults.get("xmax", 0.68)), 0.01)
with roi_col_3:
    roi_ymin = st.slider("ROI y minimum", 0.40, 0.90, float(roi_defaults.get("ymin", 0.70)), 0.01)
with roi_col_4:
    roi_ymax = st.slider("ROI y maximum", 0.75, 1.0, float(roi_defaults.get("ymax", 0.96)), 0.01)

visual_col_1, visual_col_2, visual_col_3 = st.columns(3)
with visual_col_1:
    visual_threshold = st.slider(
        "Visual detection threshold", 0.20, 0.90, float(defaults.get("visual_threshold", 0.50)), 0.01
    )
with visual_col_2:
    save_score_margin = st.slider(
        "Diagnostic save margin", 0.0, 0.25, float(defaults.get("save_score_margin", 0.08)), 0.01,
        help="Frames slightly below the threshold are also saved for false-negative inspection.",
    )
with visual_col_3:
    jpeg_quality = st.slider("Diagnostic JPEG quality", 60, 100, 85, 5)

st.subheader("Temporal event grouping")
group_col_1, group_col_2, group_col_3, group_col_4 = st.columns(4)
with group_col_1:
    max_gap_seconds = st.slider(
        "Maximum gap between positives", 0.25, 3.0, float(defaults.get("max_gap_seconds", 0.80)), 0.05
    )
with group_col_2:
    min_positive_frames = st.number_input(
        "Minimum positive samples", 1, 10, int(defaults.get("min_positive_frames", 2)), 1
    )
with group_col_3:
    event_padding = st.slider(
        "Event padding (seconds)", 0.0, 2.0, float(defaults.get("event_padding_seconds", 0.25)), 0.05
    )
with group_col_4:
    strong_single_threshold = st.slider(
        "Strong single-frame threshold", 0.40, 0.95,
        float(defaults.get("strong_single_frame_threshold", 0.72)), 0.01,
    )

st.subheader("Optional OCR and participant matching")
ocr_col_1, ocr_col_2, ocr_col_3 = st.columns(3)
with ocr_col_1:
    run_ocr = st.toggle("Run EasyOCR on detected events", value=False)
with ocr_col_2:
    ocr_gpu = st.toggle("Use GPU for OCR", value=False, disabled=not run_ocr)
with ocr_col_3:
    ocr_name_similarity = st.slider(
        "Minimum participant-name similarity", 0.40, 1.0,
        float(defaults.get("ocr_min_name_similarity", 0.68)), 0.01,
        disabled=not run_ocr,
    )

if run_ocr and not easyocr_available():
    st.warning(
        "EasyOCR is not installed. Visual detection remains fully available. Install the optional "
        "OCR dependencies with `pip install -e \".[ocr]\"`, restart Streamlit, and rerun OCR."
    )

if roi_xmax <= roi_xmin or roi_ymax <= roi_ymin:
    st.error("The ROI maximum coordinates must exceed the minimum coordinates.")
    st.stop()

scope_token = selected_part_id or "all_parts"
mode_token = (
    f"dev_{float(start_offset):.1f}_{float(development_minutes):.1f}min"
    if development_mode
    else "full"
)
run_parameters = {
    "episode_id": str(episode_id),
    "scope": scope_token,
    "mode": mode_token,
    "sample_interval_seconds": float(sample_interval),
    "roi": {"xmin": roi_xmin, "ymin": roi_ymin, "xmax": roi_xmax, "ymax": roi_ymax},
    "visual_threshold": float(visual_threshold),
    "save_score_margin": float(save_score_margin),
    "max_gap_seconds": float(max_gap_seconds),
    "min_positive_frames": int(min_positive_frames),
    "event_padding_seconds": float(event_padding),
    "strong_single_frame_threshold": float(strong_single_threshold),
    "ocr_requested": bool(run_ocr),
    "ocr_gpu": bool(ocr_gpu),
    "ocr_min_name_similarity": float(ocr_name_similarity),
}
config_token = parameter_fingerprint(run_parameters)
run_id = f"{episode_id}__{scope_token}__{mode_token}__ltcfg{config_token}"
frame_table_path = FRAME_RESULT_DIR / f"{run_id}_lower_third_frames.parquet"
event_table_path = EVENT_DIR / f"{run_id}_lower_third_events.parquet"
manifest_path = event_table_path.with_suffix(".manifest.json")
run_image_dir = IMAGE_DIR / run_id

with st.expander("Execution details", expanded=False):
    st.dataframe(selected_parts[["part_id", "part_order", "selected_video_path"]], width="stretch")
    st.write(f"**Frame-level results:** `{frame_table_path}`")
    st.write(f"**Event table:** `{event_table_path}`")
    st.json(run_parameters)


def _compute_offsets(all_parts: pd.DataFrame) -> dict[str, float]:
    offsets: dict[str, float] = {}
    offset = 0.0
    for _, part in all_parts.sort_values("part_order").iterrows():
        part_id = str(part["part_id"])
        offsets[part_id] = offset
        offset += get_video_duration_seconds(part["selected_video_path"])
    return offsets


def _render_results(frame_results: pd.DataFrame, events: pd.DataFrame) -> None:
    if frame_results.empty:
        st.info("No frame-level lower-third results are available.")
        return
    success = frame_results["analysis_success"].fillna(False).astype(bool)
    detected = frame_results.get("lower_third_detected", pd.Series(False, index=frame_results.index))
    detected = detected.fillna(False).astype(bool)
    matched = (
        int(events["matched_participant"].notna().sum())
        if not events.empty and "matched_participant" in events.columns
        else 0
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("Samples analyzed", f"{int(success.sum()):,}")
    metric_columns[1].metric("Positive samples", f"{int(detected.sum()):,}")
    metric_columns[2].metric("Temporal events", f"{len(events):,}")
    event_duration = (
        pd.to_numeric(events["duration"], errors="coerce").fillna(0).sum()
        if not events.empty and "duration" in events.columns
        else 0.0
    )
    metric_columns[3].metric("Estimated label time", f"{event_duration:.1f}s")
    metric_columns[4].metric("OCR name matches", f"{matched:,}")

    diagnostics_tab, gallery_tab, tables_tab, errors_tab = st.tabs(
        ["Diagnostics", "Event gallery", "Output tables", "Errors"]
    )
    with diagnostics_tab:
        figure = plot_lower_third_timeline(frame_results)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
        chart_col_1, chart_col_2 = st.columns(2)
        with chart_col_1:
            figure = plot_lower_third_score_distribution(frame_results)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
        with chart_col_2:
            figure = plot_lower_third_event_durations(events)
            if figure is not None:
                st.plotly_chart(figure, width="stretch")

    with gallery_tab:
        if events.empty:
            st.info("No grouped lower-third events were detected with the current parameters.")
        else:
            ordered = events.sort_values("max_visual_score", ascending=False)
            preview_count = st.slider(
                "Events to preview", 1, min(40, len(ordered)), min(12, len(ordered)),
                key=f"lt_gallery_{event_table_path.stem}",
            )
            for _, event in ordered.head(preview_count).iterrows():
                st.markdown(
                    f"**{event['lower_third_event_id']}** · score `{event['max_visual_score']:.3f}` · "
                    f"{event['part_start_time']:.2f}–{event['part_end_time']:.2f}s"
                )
                image_col_1, image_col_2, detail_col = st.columns([2, 2, 1])
                with image_col_1:
                    full_path = event.get("representative_full_frame_path")
                    if isinstance(full_path, str) and Path(full_path).exists():
                        st.image(full_path, caption="Representative full frame", width="stretch")
                with image_col_2:
                    crop_path = event.get("representative_roi_crop_path")
                    if isinstance(crop_path, str) and Path(crop_path).exists():
                        st.image(crop_path, caption="Lower-third ROI", width="stretch")
                with detail_col:
                    st.metric("Positive samples", int(event["positive_frame_count"]))
                    if pd.notna(event.get("matched_participant")):
                        st.success(str(event["matched_participant"]))
                    if pd.notna(event.get("ocr_raw_text")):
                        st.caption(str(event["ocr_raw_text"]))

    with tables_tab:
        st.markdown("#### Temporal lower-third events")
        st.dataframe(events, width="stretch")
        st.markdown("#### Frame-level visual features")
        st.dataframe(frame_results, width="stretch")

    with errors_tab:
        errors = frame_results[frame_results["error"].notna()].copy()
        ocr_errors = events[events["ocr_error"].notna()].copy() if not events.empty else pd.DataFrame()
        if errors.empty and ocr_errors.empty:
            st.success("No scan or OCR errors were recorded.")
        if not errors.empty:
            st.dataframe(errors, width="stretch")
        if not ocr_errors.empty:
            st.dataframe(ocr_errors, width="stretch")

    st.info(
        "These events are independent temporal signals. Select this event table in Commentary "
        "Candidates to promote overlapping scenes, including short comments merged inside a longer shot."
    )
    st.page_link("pages/5_Commentary_Candidates.py", label="Continue to Commentary Candidates", icon="🎯")


if st.button("Scan lower-thirds", type="primary"):
    offsets = _compute_offsets(selected_all_parts)
    frame_tables: list[pd.DataFrame] = []
    total_parts = len(selected_parts)
    progress = st.progress(0, text="Preparing lower-third scan...")
    current = st.empty()

    visual_config = LowerThirdVisualConfig(
        roi=LowerThirdROI(xmin=roi_xmin, ymin=roi_ymin, xmax=roi_xmax, ymax=roi_ymax),
        visual_threshold=float(visual_threshold),
    )

    with st.status("Scanning video for lower-thirds...", expanded=True) as status:
        try:
            for part_number, (_, part) in enumerate(selected_parts.iterrows(), start=1):
                part_id = str(part["part_id"])
                video_path = Path(part["selected_video_path"])
                part_duration = get_video_duration_seconds(video_path)
                scan_start = float(start_offset) if development_mode else 0.0
                scan_end = (
                    min(part_duration, scan_start + float(development_minutes) * 60.0)
                    if development_mode
                    else part_duration
                )

                def update_part_progress(done: int, total: int, label: str) -> None:
                    within_part = done / max(total, 1)
                    overall = ((part_number - 1) + within_part) / max(total_parts, 1)
                    progress.progress(int(overall * 100), text=f"Scanning {part_id}: {done:,}/{total:,}")
                    current.caption(label)

                frame_table = scan_lower_thirds_in_video(
                    video_path,
                    episode_id=str(episode_id),
                    part_id=part_id,
                    part_order=int(part["part_order"]),
                    global_offset_seconds=float(offsets.get(part_id, 0.0)),
                    sample_interval_seconds=float(sample_interval),
                    start_time_seconds=scan_start,
                    end_time_seconds=scan_end,
                    visual_config=visual_config,
                    output_dir=run_image_dir,
                    save_score_margin=float(save_score_margin),
                    jpeg_quality=int(jpeg_quality),
                    progress_callback=update_part_progress,
                )
                frame_tables.append(frame_table)

            frame_results = pd.concat(frame_tables, ignore_index=True) if frame_tables else pd.DataFrame()
            events = group_lower_third_events(
                frame_results,
                max_gap_seconds=float(max_gap_seconds),
                min_positive_frames=int(min_positive_frames),
                event_padding_seconds=float(event_padding),
                strong_single_frame_threshold=float(strong_single_threshold),
            )

            if run_ocr and easyocr_available() and not events.empty:
                participants = (
                    active_cast[active_cast["episode_id"].astype(str) == str(episode_id)]["participant"]
                    .dropna().astype(str).unique().tolist()
                )
                events = annotate_events_with_ocr(
                    events,
                    participants=participants,
                    reader=EasyOCRReader(EasyOCRConfig(gpu=bool(ocr_gpu))),
                    minimum_name_similarity=float(ocr_name_similarity),
                )

            save_table(frame_results, frame_table_path)
            save_table(events, event_table_path)
            write_manifest(
                manifest_path,
                {
                    "stage": "lower_third_detection",
                    "run_id": run_id,
                    "frame_table": str(frame_table_path),
                    "event_table": str(event_table_path),
                    "image_directory": str(run_image_dir),
                    "sample_count": int(len(frame_results)),
                    "event_count": int(len(events)),
                    "parameters": run_parameters,
                },
            )
            st.session_state["lower_third_last_output"] = str(event_table_path)
        except Exception as exc:
            status.update(label="Lower-third scan failed.", state="error", expanded=True)
            st.exception(exc)
            st.stop()
        status.update(label="Lower-third scan complete.", state="complete", expanded=False)
    progress.progress(100, text="Lower-third scan complete.")
    st.success("The scan, events, diagnostics, and parameters were saved for later review.")

st.divider()
st.subheader("Saved lower-third analyses")
history = list_artifacts(EVENT_DIR, "*_lower_third_events.parquet")
if not history:
    st.info("No saved lower-third event analysis is available yet.")
else:
    preferred = st.session_state.get("lower_third_last_output")
    selected_event_path = st.selectbox(
        "Analysis to visualize",
        history,
        index=preferred_artifact_index(history, preferred),
        format_func=lambda path: artifact_label(path, EVENT_DIR),
        key="lower_third_history_selector",
    )
    selected_manifest = load_manifest(selected_event_path.with_suffix(".manifest.json"))
    if selected_manifest:
        with st.expander("Saved execution parameters", expanded=False):
            st.json(selected_manifest)
    selected_run_id = selected_event_path.name.removesuffix("_lower_third_events.parquet")
    selected_frame_path = FRAME_RESULT_DIR / f"{selected_run_id}_lower_third_frames.parquet"
    selected_events = load_table(selected_event_path)
    selected_frames = load_table(selected_frame_path)
    st.caption(f"Loaded `{selected_event_path}`")
    _render_results(selected_frames, selected_events)
