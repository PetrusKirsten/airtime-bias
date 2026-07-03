import math


def face_area_ratio(face: dict) -> float:
    """Return normalized face area when face width/height are normalized to frame size."""
    return max(0.0, face.get("width", 0.0)) * max(0.0, face.get("height", 0.0))


def face_center_distance(face: dict) -> float:
    """Return Euclidean distance between the face center and the frame center."""
    cx = face.get("xmin", 0.0) + face.get("width", 0.0) / 2
    cy = face.get("ymin", 0.0) + face.get("height", 0.0) / 2
    return math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)


def score_commentary_candidate(
    faces: list[dict],
    duration: float,
    min_duration_seconds: float = 2.0,
    min_face_area_ratio: float = 0.03,
    max_center_distance: float = 0.35,
) -> dict:
    """Score whether a segment resembles a participant commentary/talking-head shot.

    This first version is intentionally heuristic. It is meant to produce confidence tiers
    and review targets, not perfect automatic labels.
    """
    if not faces:
        return {
            "commentary_score": 0.0,
            "review_required": False,
            "is_candidate": False,
            "face_count": 0,
        }

    main_face = max(faces, key=face_area_ratio)
    one_face_score = 1.0 if len(faces) == 1 else 0.4
    duration_score = min(duration / min_duration_seconds, 1.0)
    area_score = min(face_area_ratio(main_face) / min_face_area_ratio, 1.0)
    center_score = max(0.0, 1.0 - face_center_distance(main_face) / max_center_distance)

    score = 0.25 * one_face_score + 0.20 * duration_score + 0.30 * area_score + 0.25 * center_score
    return {
        "commentary_score": round(float(score), 4),
        "review_required": 0.50 <= score < 0.70,
        "is_candidate": score >= 0.70,
        "main_face_area_ratio": round(float(face_area_ratio(main_face)), 4),
        "main_face_center_distance": round(float(face_center_distance(main_face)), 4),
        "face_count": len(faces),
    }
