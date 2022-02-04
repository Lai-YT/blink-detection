import cv2
import dlib
import imutils
import time
from imutils import face_utils

from blink_detector import (
    AntiNoiseBlinkDetector,
    BlinkDetector,
    draw_landmarks_used_by_blink_detector,
)
from util.color import RED


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3

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
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        faces = detector(gray)

        # loop over the face detections
        for face in faces:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, face)
            landmarks = face_utils.shape_to_np(shape)

            ratio = BlinkDetector.get_average_eye_aspect_ratio(landmarks)
            f.write(f"{ratio:.3f}\n")

            if blink_detector.detect_blink(landmarks):
                blink_count += 1
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
