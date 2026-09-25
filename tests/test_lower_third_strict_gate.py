import cv2
import numpy as np

from airtime_bias.lower_third.visual_detection import analyze_lower_third_frame


def _base_frame() -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (70, 80, 95)
    return frame


def _participant_label() -> np.ndarray:
    frame = _base_frame()
    cv2.rectangle(frame, (55, 520), (815, 660), (72, 112, 170), -1)
    cv2.circle(frame, (110, 590), 46, (95, 145, 205), 7)
    cv2.putText(
        frame,
        "LARISSA, 22 @mc13_larissa",
        (175, 585),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (245, 245, 245),
        3,
    )
    cv2.putText(
        frame,
        "MONTADORA VEICULAR - SAO BERNARDO/SP",
        (175, 630),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
    )
    return frame


def _dark_sponsor_board() -> np.ndarray:
    frame = _base_frame()
    cv2.rectangle(frame, (45, 500), (570, 680), (35, 35, 42), -1)
    cv2.rectangle(frame, (45, 500), (570, 515), (45, 125, 205), -1)
    cv2.putText(
        frame,
        "NOMAD",
        (150, 610),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 220, 245),
        5,
    )
    return frame


def _white_sponsor_logo() -> np.ndarray:
    frame = _base_frame()
    cv2.rectangle(frame, (25, 575), (335, 665), (235, 235, 235), -1)
    cv2.putText(
        frame,
        "LE CORDON BLEU",
        (45, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (55, 55, 75),
        2,
    )
    cv2.putText(
        frame,
        "www.cordonbleu.edu",
        (45, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (55, 55, 75),
        1,
    )
    return frame


def _red_set_panel() -> np.ndarray:
    frame = _base_frame()
    cv2.rectangle(frame, (35, 500), (565, 705), (35, 35, 190), -1)
    for x in range(80, 520, 85):
        cv2.rectangle(frame, (x, 545), (x + 45, 590), (215, 205, 145), -1)
    return frame


def test_strict_detector_accepts_participant_label():
    result = analyze_lower_third_frame(_participant_label())
    assert result["visual_gate_pass"] is True
    assert result["lower_third_detected"] is True
    assert result["copper_horizontal_continuity"] >= 0.45
    assert result["text_two_line_support"] >= 0.50


def test_strict_detector_rejects_dark_sponsor_board():
    result = analyze_lower_third_frame(_dark_sponsor_board())
    assert result["visual_gate_pass"] is False
    assert result["lower_third_detected"] is False


def test_strict_detector_rejects_white_sponsor_logo():
    result = analyze_lower_third_frame(_white_sponsor_logo())
    assert result["visual_gate_pass"] is False
    assert result["lower_third_detected"] is False


def test_strict_detector_rejects_red_set_panel():
    result = analyze_lower_third_frame(_red_set_panel())
    assert result["visual_gate_pass"] is False
    assert result["lower_third_detected"] is False
