import pytest

from airtime_bias.video.frame_sampling import get_sampling_timestamps


def test_short_scene_samples_at_normalized_inner_positions():
    timestamps = get_sampling_timestamps(
        start_time=10.0,
        end_time=11.0,
        edge_margin_seconds=0.25,
    )

    assert timestamps["start"] == pytest.approx(10.30)
    assert timestamps["middle"] == pytest.approx(10.50)
    assert timestamps["end"] == pytest.approx(10.70)


def test_two_second_scene_uses_short_scene_strategy():
    timestamps = get_sampling_timestamps(
        start_time=20.0,
        end_time=22.0,
        edge_margin_seconds=0.25,
    )

    assert timestamps["start"] == pytest.approx(20.60)
    assert timestamps["middle"] == pytest.approx(21.00)
    assert timestamps["end"] == pytest.approx(21.40)


def test_long_scene_keeps_fixed_edge_margin():
    timestamps = get_sampling_timestamps(
        start_time=30.0,
        end_time=34.0,
        edge_margin_seconds=0.25,
    )

    assert timestamps["start"] == pytest.approx(30.25)
    assert timestamps["middle"] == pytest.approx(32.00)
    assert timestamps["end"] == pytest.approx(33.75)


def test_sampling_preserves_requested_positions_only():
    timestamps = get_sampling_timestamps(
        start_time=0.0,
        end_time=1.5,
        frame_positions=("middle",),
    )

    assert set(timestamps) == {"middle"}
    assert timestamps["middle"] == pytest.approx(0.75)


def test_invalid_short_scene_fraction_is_rejected():
    with pytest.raises(ValueError):
        get_sampling_timestamps(
            start_time=0.0,
            end_time=1.0,
            short_scene_edge_fraction=0.50,
        )
