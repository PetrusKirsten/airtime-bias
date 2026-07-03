import numpy as np


def placeholder_embedding(frame) -> np.ndarray:
    """Temporary placeholder until InsightFace/DeepFace integration.

    This keeps the software architecture testable before choosing the final recognition backend.
    """
    return np.zeros(512, dtype=float)
