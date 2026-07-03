# Methodology

Airtime Bias is a Streamlit-first, human-in-the-loop pipeline for measuring narrative airtime in reality TV.

The initial MVP focuses on individual participant commentary or talking-head segments. These scenes are used as the first analytical target because they are visually constrained and editorially meaningful.

## Pipeline overview

1. **Episode setup**  
   The user registers local episode metadata, including episode ID, local video path, active participants and eliminated participant.

2. **Scene detection**  
   The video is divided into candidate segments using shot/scene boundary detection.

3. **Frame sampling**  
   A small number of frames is sampled per segment to reduce processing cost.

4. **Commentary candidate scoring**  
   Each segment is scored using visual heuristics such as face count, dominant face size, face centering and segment duration.

5. **Participant identity matching**  
   Candidate segments can be matched against local participant reference images using face embeddings.

6. **Confidence-based review**  
   High-confidence segments can be accepted automatically. Low-confidence or uncertain cases are routed to manual review.

7. **Exposure metrics**  
   The pipeline aggregates commentary/talking-head airtime by participant and episode.

8. **Elimination comparison**  
   The eliminated participant is compared with the rest of the cast and with their own previous exposure when enough episodes are available.

## Interpretation

The project should be interpreted as exploratory media analytics. Airtime metrics can reveal patterns of narrative emphasis, but they do not directly prove producer intent, causality or manipulation.
