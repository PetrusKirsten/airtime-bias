import numpy as np


def detect_faces_mediapipe(frame: np.ndarray) -> list[dict]:
    """Detect faces in a frame using MediaPipe.

    Returns normalized bounding boxes and confidence scores.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("Install mediapipe with: pip install mediapipe") from exc

    mp_face_detection = mp.solutions.face_detection
    rgb = frame[:, :, ::-1]
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        result = detector.process(rgb)

    faces = []
    if not result.detections:
        return faces

    for detection in result.detections:
        bbox = detection.location_data.relative_bounding_box
        faces.append(
            {
                "xmin": bbox.xmin,
                "ymin": bbox.ymin,
                "width": bbox.width,
                "height": bbox.height,
                "confidence": float(detection.score[0]),
            }
        )
    return faces
