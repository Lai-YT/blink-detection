import cv2
import dlib
import imutils
import time
from functools import partial
from operator import methodcaller
from typing import Optional

from imutils import face_utils

from blink_detector import (
    AntiNoiseBlinkDetector,
    BlinkDetector,
)
from util.color import RED
from util.faceplots import (
    draw_landmarks_used_by_blink_detector,
    mark_face,
)


def clamp(value: float, v_min: float, v_max: float) -> float:
    """Clamps the value into the range [v_min, v_max].

    e.g., _clamp(50, 20, 40) returns 40.
    v_min should be less or equal to v_max. (v_min <= v_max)
    """
    if not v_min < v_max:
        raise ValueError("v_min is the lower bound, which should be smaller than v_max")

    if value > v_max:
        value = v_max
    elif value < v_min:
        value = v_min
    return value


def get_biggest_face(faces: dlib.rectangles) -> Optional[dlib.rectangle]:
    """Returns the face with the biggest area. None if the input faces is empty."""
    # faces are compared through the area method
    return max(faces, default=None, key=methodcaller("area"))


def get_face_area_frame(frame):
    """Returns the main face area if the frame contains any face.

    Note that the area is about 2 times enlarged (1.4 x width and 1.4 x height)
    to make sure the face isn't cut.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face = get_biggest_face(faces)
    if face is not None:
        # extract the face area from the frame
        fx, fy, fw, fh = face_utils.rect_to_bb(face)
        ih, iw, _ = frame.shape

        clamp_height = partial(clamp, v_min=0, v_max=ih)
        clamp_width = partial(clamp, v_min=0, v_max=iw)
        # NOTE: this makes a view, not copy
        frame = frame[int(clamp_height(fy-0.2*fh)):int(clamp_height(fy+1.2*fh)),
                      int(clamp_width(fx-0.2*fw)):int(clamp_width(fx+1.2*fw))]
    return frame


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3  # the face area mode consumes more computaional time,
                          # 2 would be more suitable

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

print("[INFO] preparing blink detector...")
blink_detector = AntiNoiseBlinkDetector(EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES)

# start the video stream
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

# initialize the total number of blinks
blink_count = 0

# EAR logging file
with open("./ratio.txt", "w+") as f:
    # loop over frames from the video stream
    while cam.isOpened():

        # grab the frame from the camera, resize
        # it, and convert it to grayscale channels
        _, frame = cam.read()

        # Uncomment the following line to process blink detection on the
        # extracted face area.
        # frame = get_face_area_frame(frame)

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        faces = detector(gray)
        face = get_biggest_face(faces)

        if face is not None:
            # determine the facial landmarks for the face area, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, face)
            landmarks = face_utils.shape_to_np(shape)

            if blink_detector.detect_blink(landmarks):
                blink_count += 1
                # the one right after an end of blink is marked
                f.write("* ")

            ratio = BlinkDetector.get_average_eye_aspect_ratio(landmarks)
            f.write(f"{ratio:.3f}\n")

            frame = draw_landmarks_used_by_blink_detector(frame, landmarks)
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            cv2.putText(frame, f"EAR: {ratio:.2f}", (450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
cam.release()
