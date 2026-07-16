# Methodology

Airtime Bias is a Streamlit-first, human-in-the-loop pipeline for measuring narrative airtime in reality TV.

The initial MVP focuses on individual participant commentary or talking-head segments. These scenes are used as the first analytical target because they are visually constrained and editorially meaningful.

## Pipeline overview

1. **Episode setup**  
   The user registers local episode metadata, including episode ID, local video parts, active participants and eliminated participant.

2. **Scene detection**  
   Each episode part is divided into raw shots using PySceneDetect. The current calibrated baseline is a content threshold of `44.0` and a minimum interval between detected cuts of `2.0` seconds. These values are treated as configurable experimental parameters rather than universal defaults.

3. **Representative frame sampling**  
   Three frames are extracted from each shot: start, middle and end. Start/end samples are shifted inward by a configurable margin to reduce transition frames and motion blur. Sampled frames remain local and are excluded from version control.

4. **Middle-frame face prefilter**  
   The middle frame is analyzed first using a reusable MediaPipe face detector. Obvious negatives, such as frames without a sufficiently large and reasonably central face, are rejected before start/end analysis. The prefilter is deliberately permissive because its main goal is reducing computation without sacrificing recall.

5. **Three-frame visual consistency analysis**  
   Shots that pass the middle-frame prefilter are evaluated across start, middle and end. Frame-level features include face count, dominant face area, distance from frame center and detection confidence.

6. **Interpretable commentary candidate score**  
   Frame features are aggregated into segment-level components:

   - face presence ratio;
   - single-face ratio;
   - dominant-face area score;
   - face-centering score;
   - face geometry stability;
   - duration score.

   The baseline score is a weighted heuristic:

   ```text
   commentary_score =
       0.30 × face_presence
     + 0.20 × single_face
     + 0.20 × face_area
     + 0.15 × centering
     + 0.10 × geometry_stability
     + 0.05 × duration
   ```

   Results are separated into three tiers: `candidate`, `review` and `reject`. These thresholds are adjustable and must be validated against manually labeled scenes.

7. **Participant identity matching**  
   Candidate segments can be matched against local participant reference images using face embeddings. Identity analysis is restricted to the reduced candidate set rather than the full episode.

8. **Confidence-based review**  
   High-confidence segments can be accepted automatically. Borderline candidates, uncertain identities and suspected false positives are routed to manual review.

9. **Exposure metrics**  
   The pipeline aggregates commentary/talking-head airtime by participant and episode.

10. **Elimination comparison**  
    The eliminated participant is compared with the rest of the cast and with their own previous exposure when enough episodes are available.

## Validation priorities

Because commentary scenes are expected to be a minority class, overall accuracy is not the primary quality metric. Calibration should prioritize:

- recall of manually identified commentary scenes;
- precision among retained candidates;
- false-negative rate;
- candidate reduction rate;
- proportion of scenes routed to review;
- error propagated to participant airtime totals.

## Interpretation

The project should be interpreted as exploratory media analytics. Airtime metrics can reveal patterns of narrative emphasis, but they do not directly prove producer intent, causality or manipulation.
