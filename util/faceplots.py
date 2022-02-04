from typing import Tuple

import cv2
from nptyping import Int, NDArray

from blink_detector import BlinkDetector
from util.color import BGR, GREEN, MAGENTA
from util.image_type import ColorImage


def mark_face(
        canvas: ColorImage,
        face: Tuple[int, int, int, int],
        landmarks: NDArray[(68, 2), Int[32]]) -> None:
    """Modifies the canvas with face area framed up and landmarks dotted.
    
    Arguments:
        canvas: The image to mark face on.
        face: Upper-left x, y coordinates of face and it's width, height.
        landmarks: (x, y) coordinates of the 68 face landmarks.
    """
    fx, fy, fw, fh = face
    cv2.rectangle(canvas, (fx, fy), (fx+fw, fy+fh), MAGENTA, 1)
    for lx, ly in landmarks:
        cv2.circle(canvas, (lx, ly), 1, GREEN, -1)


def draw_landmarks_used_by_blink_detector(
        canvas: ColorImage,
        landmarks: NDArray[(68, 2), Int[32]],
        color: BGR = GREEN) -> ColorImage:
    """Returns the canvas with the eyes' contours.

    Arguments:
        canvas: The image to draw on, it'll be copied.
        landmarks: (x, y) coordinates of the 68 face landmarks.
        color: Color of the lines, green (0, 255, 0) in default.
    """
    canvas_: ColorImage = canvas.copy()

    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    for start, end in (BlinkDetector.LEFT_EYE_START_END_IDXS, BlinkDetector.RIGHT_EYE_START_END_IDXS):
        hull = cv2.convexHull(landmarks[start:end])
        cv2.drawContours(canvas_, [hull], -1, color, 1)

    # make lines transparent
    canvas_ = cv2.addWeighted(canvas_, 0.4, canvas, 0.6, 0)
    return canvas_
