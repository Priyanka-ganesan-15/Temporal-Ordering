import numpy as np

from chronologic.ordering.text_direction import temporal_direction_score


def test_temporal_direction_score_prefers_forward_alignment() -> None:
    # Frame 0 aligns with start, frame 1 with middle, frame 2 with end.
    similarities = {
        "start": np.array([0.9, 0.3, 0.1], dtype=np.float32),
        "middle": np.array([0.2, 0.95, 0.2], dtype=np.float32),
        "end": np.array([0.1, 0.4, 0.92], dtype=np.float32),
    }

    forward = temporal_direction_score([0, 1, 2], similarities)
    reverse = temporal_direction_score([2, 1, 0], similarities)

    assert forward > reverse
