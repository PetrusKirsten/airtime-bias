# Methodology

Airtime Bias is a Streamlit-first, human-in-the-loop pipeline for measuring narrative airtime in reality TV.

The initial MVP focuses on individual participant commentary or talking-head segments. These scenes are used as the first analytical target because they are visually constrained and editorially meaningful.

## Pipeline overview

1. **Episode setup**  
   The user registers local episode metadata, including episode ID, local video parts, active participants and eliminated participant.

2. **High-sensitivity scene detection**  
   Each episode part is divided into raw shots using PySceneDetect. The current working baseline uses a content threshold of `44.0` and a configurable minimum interval between cuts. After manual review revealed genuine commentary moments below two seconds, the default minimum interval was reduced from `2.0` to `1.0` second. Saved experiments remain available so `0.5`, `1.0` and longer settings can be compared rather than treated as universal constants.

3. **Representative frame sampling**  
   Three frames are extracted from each shot: start, middle and end. Start/end samples are shifted inward by a configurable margin to reduce transition frames and motion blur. Sampled frames remain local and are excluded from version control.

4. **Middle-frame face prefilter**  
   The middle frame is analyzed first using a reusable MediaPipe face detector. The current target is usually a single dominant face; therefore, the default prefilter allows at most one detected face while keeping this parameter adjustable for recall checks.

5. **Three-frame visual consistency analysis**  
   Shots that pass the middle-frame prefilter are evaluated across start, middle and end. Frame-level features include face count, dominant face area, distance from frame center and detection confidence.

6. **Face-based commentary candidate score**  
   Frame features are aggregated into segment-level components:

   - face presence ratio;
   - single-face ratio;
   - dominant-face area score;
   - face-centering score;
   - face geometry stability;
   - duration diagnostic.

   The current baseline score is:

   ```text
   face_commentary_score =
       0.30 × face_presence
     + 0.25 × single_face
     + 0.20 × face_area
     + 0.15 × centering
     + 0.10 × geometry_stability
     + 0.00 × duration
   ```

   Duration remains in the output for analysis but has zero default weight, preventing a genuine sub-two-second comment from being demoted solely because it is brief.

7. **Independent participant lower-third scan**  
   The lower-left area of the video is sampled at a fixed temporal interval independently of scene segmentation. An interpretable OpenCV detector combines warm rectangular structure, edge density, bright text-like components, row coherence and circular-logo evidence. Positive samples are grouped into temporal lower-third events. This route can flag a commentary moment even when it was merged into a longer raw shot.

8. **Optional OCR and active-cast matching**  
   OCR is applied only to representative positive lower-third crops. The extracted text is normalized and matched against the closed list of participants active in the episode. OCR is optional: the visual lower-third detector and event timeline remain fully functional without it.

9. **Signal fusion**  
   Face candidates and lower-third events are aligned by episode, part and time overlap. The output preserves the original face score and records whether evidence came from `face`, `lower_third` or `face+lower_third`. A strong visual label or matched participant name may promote a face-based reject into the review or candidate tier.

10. **Persisted manual review**  
    Retained scenes are classified with two separate concepts:

    - scene type: `commentary`, `participant_closeup`, `other`, `uncertain` or `pending`;
    - participant identity, when known.

    This separation allows a non-commentary close-up to become a useful identity example without contaminating commentary airtime. Review-queue refreshes preserve existing decisions by `segment_id`.

11. **Participant identity matching**  
    Named commentary and participant-close-up frames can be exported as a labeled identity dataset. A future recognition model should use separate development and validation examples to avoid evaluating on its own references.

12. **Exposure metrics and episode events**  
    The pipeline aggregates commentary/talking-head airtime by participant and episode. The event schema is intended to expand later to challenge wins, eliminations and participant-focused trajectory/backstory packages.

13. **Elimination and narrative-event comparison**  
    The eliminated participant can be compared with the remaining cast, their own previous exposure, challenge outcomes and the occurrence of participant-focused narrative packages when enough episodes have been annotated.

## Validation priorities

Because commentary scenes are expected to be a minority class, overall accuracy is not the primary quality metric. Calibration should prioritize:

- recall of manually identified commentary scenes, including sub-two-second cases;
- precision among retained candidates;
- false-negative rate in both face and lower-third routes;
- lower-third event precision and temporal coverage;
- OCR participant-match accuracy when OCR is enabled;
- candidate reduction rate;
- proportion of scenes routed to review;
- error propagated to participant airtime totals.

## Interpretation

The project should be interpreted as exploratory media analytics. Airtime metrics and event associations can reveal patterns of narrative emphasis, but they do not directly prove producer intent, causality or manipulation.
