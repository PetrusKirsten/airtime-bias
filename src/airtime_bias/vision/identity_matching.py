import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_identity(embedding: np.ndarray, gallery: dict[str, np.ndarray], threshold: float = 0.6) -> dict:
    if not gallery:
        return {"identity": "unknown", "confidence": 0.0}
    scores = {name: cosine_similarity(embedding, ref) for name, ref in gallery.items()}
    best_name = max(scores, key=scores.get)
    confidence = scores[best_name]
    if confidence < threshold:
        return {"identity": "unknown", "confidence": confidence}
    return {"identity": best_name, "confidence": confidence}
