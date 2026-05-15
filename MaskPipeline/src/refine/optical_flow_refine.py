from __future__ import annotations

import cv2
import numpy as np


def refine_with_optical_flow(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    if previous_frame is None:
        return candidate_mask

    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=5,
        mask=candidate_mask.astype(np.uint8),
    )
    if points is None or len(points) < 8:
        return candidate_mask

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None)
    if next_points is None or status is None:
        return candidate_mask

    valid_prev = points[status.flatten() == 1]
    valid_next = next_points[status.flatten() == 1]
    if len(valid_prev) < 8:
        return candidate_mask

    motion = np.linalg.norm((valid_next - valid_prev).reshape(-1, 2), axis=1)
    dynamic = np.zeros_like(candidate_mask)
    for point, magnitude in zip(valid_prev, motion):
        if float(magnitude) >= 1.0:
            x, y = point.ravel()
            cv2.circle(dynamic, (int(round(x)), int(round(y))), 10, 255, thickness=-1)
    return cv2.bitwise_and(candidate_mask.astype(np.uint8), dynamic.astype(np.uint8))
