# Airtime Bias

**Airtime Bias** is an interactive computer vision and media analytics project for investigating how reality TV distributes narrative exposure among contestants.

The project starts from a focused question:

> Do eliminated contestants receive a different pattern of narrative airtime in the episode they leave?

Rather than claiming to predict eliminations perfectly, Airtime Bias treats screen exposure as an exploratory editorial signal. The goal is to build a transparent, human-in-the-loop pipeline that detects participant commentary/talking-head segments, identifies the featured contestant, measures exposure, and validates uncertainty.

## Why this project exists

Reality shows are not only competitions; they are edited narratives. Contestants can be emphasized, hidden, foreshadowed or reframed through repeated direct-to-camera commentary, reaction shots and other forms of curated airtime.

Airtime Bias uses computer vision to ask a quantitative media question:

> Can we measure whether a contestant's narrative exposure changes near elimination?

The initial MVP focuses on individual commentary/talking-head segments because they are more constrained than full-episode recognition:

- usually one main visible participant;
- relatively stable framing;
- larger and more centered face;
- limited background variation;
- high narrative/editorial value;
- easier manual validation.

## What the project is not

Airtime Bias is **not** intended to be:

- a perfect elimination prediction model;
- an official analysis of any TV show;
- a tool for redistributing copyrighted video;
- a claim about producer intent;
- a fully automated biometric system without validation.

It is an exploratory portfolio project combining computer vision, data engineering, data science and media analytics.

## Core workflow

1. Register local episode metadata.
2. Detect scene boundaries.
3. Sample frames from each segment.
4. Score candidate commentary/talking-head segments.
5. Identify the featured participant using local reference images.
6. Flag uncertain predictions for review.
7. Generate narrative exposure metrics.
8. Compare eliminated and non-eliminated contestants.
9. Validate the pipeline and document limitations.
10. Present the results through an interactive Streamlit app.

## Streamlit app pages

The application is designed as the primary interface for the project:

| Page | Purpose |
|---|---|
| Home | Project overview and research question |
| Episode Setup | Register local videos and episode metadata |
| Scene Detection | Run scene boundary detection |
| Commentary Candidates | Score likely commentary/talking-head segments |
| Participant Review | Review predicted identities and uncertain segments |
| Exposure Analysis | Explore participant exposure metrics |
| Pipeline Validation | Measure detection and identity quality |
| Story Mode | Portfolio-facing narrative and interpretation |

## Repository structure

```text
airtime-bias/
├── app/
│   ├── Home.py
│   └── pages/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── references/
│   ├── interim/
│   ├── processed/
│   └── validation/
├── docs/
├── reports/
├── src/
│   └── airtime_bias/
│       ├── commentary/
│       ├── features/
│       ├── io/
│       ├── validation/
│       ├── video/
│       ├── vision/
│       └── visualization/
└── tests/
```

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Run the app

```bash
streamlit run app/Home.py
```

## Local data policy

The following files should stay local and are ignored by Git:

- TV episodes;
- extracted clips;
- extracted frames;
- participant reference images;
- facial embeddings;
- processed media artifacts;
- validation files containing identifiable examples.

Public outputs should prioritize:

- aggregated tables;
- plots;
- methodology;
- code;
- synthetic examples;
- anonymized examples when appropriate.

## Planned MVP

The first working version should process one episode end-to-end:

1. register episode metadata;
2. detect scenes;
3. compute commentary-candidate scores;
4. assign participant identities where confidence is high;
5. allow manual review of uncertain segments;
6. produce exposure metrics by participant;
7. compare the eliminated contestant against the episode cast.

## Key metrics

Initial analytical metrics include:

- `commentary_time_total`;
- `commentary_segments_count`;
- `avg_commentary_segment_duration`;
- `median_commentary_segment_duration`;
- `commentary_time_share`;
- `last_third_commentary_time`;
- `delta_vs_previous_episode`;
- `delta_vs_personal_average`;
- `zscore_vs_cast_in_episode`;
- `was_eliminated`;
- `confidence_mean`;
- `review_required_rate`.

## Ethics and limitations

Airtime Bias is designed for educational and portfolio use. It should not redistribute copyrighted video, publish unnecessary face images, expose facial embeddings or imply affiliation with any show, broadcaster or streaming platform.

The word **bias** is used as an investigative framing: it refers to measurable asymmetries in airtime allocation, not proof of unfairness, intent or manipulation.

See:

- [`docs/ethics.md`](docs/ethics.md)
- [`docs/limitations.md`](docs/limitations.md)
- [`docs/methodology.md`](docs/methodology.md)

## Suggested first Git commit

```bash
git init
git add .
git commit -m "Initial Airtime Bias project structure"
```
