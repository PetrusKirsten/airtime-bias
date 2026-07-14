from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from airtime_bias.commentary.candidate_detection import (
    CommentaryScoreConfig,
    build_commentary_candidate_table,
)
from airtime_bias.features.visual_face_features import extract_face_features_from_samples
from airtime_bias.io.loaders import load_config, load_table
from airtime_bias.io.paths import INTERIM_DIR, PROCESSED_DIR, ensure_project_dirs
from airtime_bias.io.writers import save_table
from airtime_bias.vision.face_detection import FaceDetectorConfig
from airtime_bias.visualization.candidate_plots import (
    plot_candidate_tier_counts,
    plot_commentary_score_distribution,
    plot_face_geometry_map,
)

ensure_project_dirs()

st.set_page_config(
    page_title="Commentary Candidates",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Commentary Candidate Detection")
st.markdown(
    """
Reduce thousands of raw shots to a smaller set of likely participant commentary scenes.
The workflow uses a **recall-oriented middle-frame prefilter**, then evaluates start,
middle, and end frames only for scenes that survive the cheap first pass.
"""
)

config = load_config()
commentary_defaults = config.get("commentary_detection", {})
face_defaults = config.get("face_detection", {})

FRAME_TABLE_DIR = INTERIM_DIR / "sampled_frames" / "tables"
FACE_FEATURE_DIR = INTERIM_DIR / "face_features"
CANDIDATE_DIR = PROCESSED_DIR / "commentary_candidates"

frame_table_files = sorted(FRAME_TABLE_DIR.glob("*_frame_samples.parquet"))
if not frame_table_files:
    st.warning("No frame sample tables found. Run Frame Sampling first.")
    st.stop()

selected_frame_table = st.selectbox(
    "Frame sample table",
    options=frame_table_files,
    format_func=lambda path: path.name,
)

frame_samples = load_table(selected_frame_table)
if frame_samples.empty:
    st.error("The selected frame sample table is empty.")
    st.stop()

required_columns = {"episode_id", "part_id", "segment_id", "frame_position", "frame_path", "success"}
missing_columns = required_columns - set(frame_samples.columns)
if missing_columns:
    st.error(f"Frame sample table is missing required columns: {sorted(missing_columns)}")
    st.stop()

frame_samples = frame_samples.copy()
frame_samples["episode_id"] = frame_samples["episode_id"].astype(str)
frame_samples["part_id"] = frame_samples["part_id"].astype(str)
frame_samples["segment_id"] = frame_samples["segment_id"].astype(str)

st.subheader("Analysis scope")
scope_col_1, scope_col_2 = st.columns(2)

with scope_col_1:
    episode_id = st.selectbox(
        "Episode",
        sorted(frame_samples["episode_id"].unique().tolist()),
    )

with scope_col_2:
    scope_mode = st.radio(
        "Scope",
        options=["All available parts", "Single part"],
        horizontal=True,
    )

selected_samples = frame_samples[frame_samples["episode_id"] == episode_id].copy()
selected_part_id: str | None = None

if scope_mode == "Single part":
    selected_part_id = st.selectbox(
        "Episode part",
        sorted(selected_samples["part_id"].unique().tolist()),
    )
    selected_samples = selected_samples[selected_samples["part_id"] == selected_part_id].copy()

st.subheader("Face detection and prefilter")
face_col_1, face_col_2, face_col_3, face_col_4 = st.columns(4)

with face_col_1:
    model_selection = st.selectbox(
        "MediaPipe model range",
        options=[0, 1],
        index=1 if int(face_defaults.get("model_selection", 1)) == 1 else 0,
        format_func=lambda value: "Full range" if value == 1 else "Short range",
    )

with face_col_2:
    min_detection_confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.95,
        value=float(face_defaults.get("min_detection_confidence", 0.50)),
        step=0.05,
    )

with face_col_3:
    use_middle_prefilter = st.toggle(
        "Use middle-frame prefilter",
        value=bool(face_defaults.get("use_middle_prefilter", True)),
        help="Analyze start/end only when the middle frame contains a plausible dominant face.",
    )

with face_col_4:
    prefilter_max_faces = st.number_input(
        "Prefilter maximum faces",
        min_value=1,
        max_value=20,
        value=int(face_defaults.get("prefilter_max_faces", 4)),
        step=1,
        disabled=not use_middle_prefilter,
    )

prefilter_col_1, prefilter_col_2 = st.columns(2)
with prefilter_col_1:
    prefilter_min_area = st.slider(
        "Prefilter minimum main-face area ratio",
        min_value=0.005,
        max_value=0.150,
        value=float(face_defaults.get("prefilter_min_face_area_ratio", 0.02)),
        step=0.005,
        format="%.3f",
        disabled=not use_middle_prefilter,
    )
with prefilter_col_2:
    prefilter_max_center = st.slider(
        "Prefilter maximum center distance",
        min_value=0.10,
        max_value=0.75,
        value=float(face_defaults.get("prefilter_max_center_distance", 0.55)),
        step=0.05,
        disabled=not use_middle_prefilter,
    )

st.caption(
    "The prefilter is deliberately permissive. Its purpose is to remove obvious negatives, "
    "not to make the final commentary decision."
)

st.subheader("Commentary score")
score_col_1, score_col_2, score_col_3, score_col_4, score_col_5 = st.columns(5)

with score_col_1:
    min_duration_seconds = st.slider(
        "Reference duration",
        min_value=0.5,
        max_value=10.0,
        value=float(commentary_defaults.get("min_duration_seconds", 2.0)),
        step=0.5,
    )
with score_col_2:
    min_face_area_ratio = st.slider(
        "Target face area ratio",
        min_value=0.005,
        max_value=0.200,
        value=float(commentary_defaults.get("min_face_area_ratio", 0.04)),
        step=0.005,
        format="%.3f",
    )
with score_col_3:
    max_center_distance = st.slider(
        "Target center distance",
        min_value=0.10,
        max_value=0.70,
        value=float(commentary_defaults.get("max_center_distance", 0.35)),
        step=0.05,
    )
with score_col_4:
    review_threshold = st.slider(
        "Review threshold",
        min_value=0.10,
        max_value=0.85,
        value=float(commentary_defaults.get("review_threshold", 0.50)),
        step=0.05,
    )
with score_col_5:
    candidate_threshold = st.slider(
        "Candidate threshold",
        min_value=0.20,
        max_value=0.95,
        value=max(
            float(commentary_defaults.get("candidate_threshold", 0.70)),
            review_threshold,
        ),
        step=0.05,
    )

if review_threshold > candidate_threshold:
    st.error("Review threshold cannot be greater than candidate threshold.")
    st.stop()

st.subheader("Execution mode")
mode_col_1, mode_col_2 = st.columns(2)
with mode_col_1:
    development_mode = st.toggle(
        "Development mode",
        value=True,
        help="Analyze only the first N scenes while calibrating the cascade and score.",
    )
with mode_col_2:
    all_segment_ids = selected_samples["segment_id"].drop_duplicates().tolist()
    max_segments = st.number_input(
        "Maximum scenes",
        min_value=1,
        max_value=max(1, len(all_segment_ids)),
        value=min(100, len(all_segment_ids)),
        step=25,
        disabled=not development_mode,
    )

if development_mode:
    selected_segment_ids = all_segment_ids[: int(max_segments)]
    samples_to_process = selected_samples[
        selected_samples["segment_id"].isin(selected_segment_ids)
    ].copy()
else:
    samples_to_process = selected_samples.copy()

scene_count = samples_to_process["segment_id"].nunique()
frame_count = len(samples_to_process)
anchor_count = scene_count
max_additional_frames = max(0, frame_count - anchor_count)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Scenes selected", f"{scene_count:,}")
metric_2.metric("Available frames", f"{frame_count:,}")
metric_3.metric("Mandatory middle pass", f"{anchor_count:,}")
metric_4.metric("Conditional frames", f"{max_additional_frames:,}")

scope_token = selected_part_id if selected_part_id is not None else "all_parts"
mode_token = f"dev{scene_count}" if development_mode else "full"
experiment_name = f"{Path(selected_frame_table).stem}__{scope_token}__{mode_token}"
face_feature_path = FACE_FEATURE_DIR / f"{experiment_name}_face_features.parquet"
candidate_path = CANDIDATE_DIR / f"{experiment_name}_commentary_candidates.parquet"

with st.expander("Execution details", expanded=False):
    st.write(f"**Input frame table:** `{selected_frame_table}`")
    st.write(f"**Face features:** `{face_feature_path}`")
    st.write(f"**Candidate table:** `{candidate_path}`")


def _render_candidate_results(
    face_features: pd.DataFrame,
    candidates: pd.DataFrame,
    source_samples: pd.DataFrame,
) -> None:
    if candidates.empty:
        st.info("No candidate rows were generated.")
        return

    tier_counts = candidates["candidate_tier"].value_counts()
    candidate_count = int(tier_counts.get("candidate", 0))
    review_count = int(tier_counts.get("review", 0))
    reject_count = int(tier_counts.get("reject", 0))
    total_scenes = len(candidates)
    retained_count = candidate_count + review_count
    reduction_rate = 1.0 - retained_count / max(total_scenes, 1)

    prefilter_passed = int(candidates["prefilter_pass"].fillna(False).sum())
    analyzed_frames = int(face_features["analysis_success"].fillna(False).sum())
    skipped_frames = int(face_features["skipped_by_prefilter"].fillna(False).sum())

    result_col_1, result_col_2, result_col_3, result_col_4 = st.columns(4)
    result_col_1.metric("High-confidence candidates", f"{candidate_count:,}")
    result_col_2.metric("Review required", f"{review_count:,}")
    result_col_3.metric("Automatic rejects", f"{reject_count:,}")
    result_col_4.metric("Review workload reduction", f"{reduction_rate:.1%}")

    detail_col_1, detail_col_2, detail_col_3 = st.columns(3)
    detail_col_1.metric("Middle prefilter passed", f"{prefilter_passed:,}/{total_scenes:,}")
    detail_col_2.metric("Frames analyzed", f"{analyzed_frames:,}")
    detail_col_3.metric("Frames skipped", f"{skipped_frames:,}")

    chart_tab, gallery_tab, table_tab, error_tab = st.tabs(
        ["Diagnostics", "Candidate gallery", "Output tables", "Errors"]
    )

    with chart_tab:
        chart_col_1, chart_col_2 = st.columns(2)
        with chart_col_1:
            fig = plot_commentary_score_distribution(candidates)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
        with chart_col_2:
            fig = plot_candidate_tier_counts(candidates)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")

        fig = plot_face_geometry_map(candidates)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

    with gallery_tab:
        tier_filter = st.multiselect(
            "Decision tiers",
            options=["candidate", "review", "reject"],
            default=["candidate", "review"],
        )
        gallery_candidates = candidates[
            candidates["candidate_tier"].isin(tier_filter)
        ].sort_values("commentary_score", ascending=False)

        if gallery_candidates.empty:
            st.info("No scenes match the selected tiers.")
        else:
            gallery_limit = st.slider(
                "Scenes to preview",
                min_value=1,
                max_value=min(40, len(gallery_candidates)),
                value=min(12, len(gallery_candidates)),
            )
            preview_candidates = gallery_candidates.head(gallery_limit)

            for _, candidate in preview_candidates.iterrows():
                segment_id = str(candidate["segment_id"])
                segment_samples = source_samples[
                    source_samples["segment_id"].astype(str) == segment_id
                ].copy()
                segment_samples["frame_position"] = segment_samples["frame_position"].astype(str)
                segment_samples = segment_samples.set_index("frame_position")

                st.markdown(
                    f"**{segment_id}** · score `{candidate['commentary_score']:.3f}` · "
                    f"tier `{candidate['candidate_tier']}` · "
                    f"face presence `{candidate['face_presence_ratio']:.0%}`"
                )
                columns = st.columns(3)
                for column, position in zip(columns, ["start", "middle", "end"]):
                    with column:
                        if position not in segment_samples.index:
                            st.caption(f"{position}: unavailable")
                            continue
                        sample = segment_samples.loc[position]
                        if isinstance(sample, pd.DataFrame):
                            sample = sample.iloc[0]
                        frame_path = sample.get("frame_path")
                        if isinstance(frame_path, str) and Path(frame_path).exists():
                            st.image(frame_path, caption=position, width="stretch")
                        else:
                            st.caption(f"{position}: file not found")

    with table_tab:
        st.markdown("#### Segment-level commentary candidates")
        st.dataframe(candidates, width="stretch")
        st.markdown("#### Frame-level face features")
        st.dataframe(face_features, width="stretch")

    with error_tab:
        errors = face_features[
            face_features["error"].notna()
            & ~face_features["skipped_by_prefilter"].fillna(False)
        ].copy()
        if errors.empty:
            st.success("No face-analysis errors were recorded.")
        else:
            st.dataframe(errors, width="stretch")


if st.button("Detect commentary candidates", type="primary"):
    progress_bar = st.progress(0, text="Preparing face analysis...")
    current_scene = st.empty()

    def update_progress(current: int, total: int, segment_id: str) -> None:
        percentage = int((current / max(total, 1)) * 100)
        progress_bar.progress(
            percentage,
            text=f"Analyzing faces: {current:,}/{total:,} scenes",
        )
        current_scene.caption(f"Current scene: `{segment_id}`")

    detector_config = FaceDetectorConfig(
        model_selection=int(model_selection),
        min_detection_confidence=float(min_detection_confidence),
    )
    score_config = CommentaryScoreConfig(
        min_duration_seconds=float(min_duration_seconds),
        min_face_area_ratio=float(min_face_area_ratio),
        max_center_distance=float(max_center_distance),
        candidate_threshold=float(candidate_threshold),
        review_threshold=float(review_threshold),
    )

    with st.status("Running commentary candidate cascade...", expanded=True) as status:
        try:
            face_features = extract_face_features_from_samples(
                frame_samples=samples_to_process,
                detector_config=detector_config,
                use_middle_prefilter=use_middle_prefilter,
                prefilter_min_face_area_ratio=float(prefilter_min_area),
                prefilter_max_center_distance=float(prefilter_max_center),
                prefilter_max_faces=int(prefilter_max_faces),
                progress_callback=update_progress,
            )
            candidates = build_commentary_candidate_table(
                frame_features=face_features,
                config=score_config,
            )
            save_table(face_features, face_feature_path)
            save_table(candidates, candidate_path)
        except Exception as exc:
            status.update(label="Commentary candidate detection failed.", state="error", expanded=True)
            st.exception(exc)
            st.stop()

        status.update(label="Commentary candidate detection complete.", state="complete", expanded=False)

    progress_bar.progress(100, text="Commentary candidate detection complete.")
    st.success(f"Saved candidate table to `{candidate_path}`")
    _render_candidate_results(face_features, candidates, samples_to_process)
else:
    existing_features = load_table(face_feature_path)
    existing_candidates = load_table(candidate_path)
    if not existing_features.empty and not existing_candidates.empty:
        st.subheader("Existing candidate detection results")
        _render_candidate_results(existing_features, existing_candidates, samples_to_process)
    else:
        st.info("Run a development batch first, inspect recall visually, then scale to a full part.")
