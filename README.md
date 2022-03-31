# Blink Detection

This study supports the *blink* module in [Lai-YT/webcam-applications](https://github.com/Lai-YT/webcam-applications).

## Purpose

We want to detect whether the user blinks or not from the video.

## Eye Aspect Ratio (EAR)

T. Soukupová and J. Čech found out that the eye aspect ratio, which is defined \
as the ratio of the height and width of the eye, changes between opened and \
closed eyes, and so we can detect the moment of blink by observing whether the \
*EAR* drops below a specific *threshould* [1]. \
But a static *threshold* fails under different situations and a dynamic one is \
hard to find. \
Instead of using fancy *SVM*, we generalize the blink detection problem to a \
**change point problem**. When a blink is made, the EAR changes dramatically and \
rapidly.

## Change Point Detection

The following GIF illustrates how a window-based change point detection works \
with the standard deviation (*STD*) as its *cost function* [2].

![Animation of change point detection via sliding window](https://www.iese.fraunhofer.de/blog/wp-content/uploads/2021/08/illustration_of_change_point_detectopn_via_sliding-window.gif)

The critical parameters are

1. `WINDOW_SIZE`: size of the window
1. `DRAMATIC_STD_CHANGE`: how dramatic the change in *STD* is to be considered \
as a change point

These parameters control how sensitive the detection is. \
In the implemention, we only look at the *STD* change between 2 successive \
points, one may look at 3 or more of them, and define the occurrence of specific \
relation to be a blink (change point). \
The following picture shows the result of a 3 minutes blink detection with \
`WINDOW_SIZE = 9` and `DRAMATIC_STD_CHANGE = 0.008`. Vertical lines are the \
blinks that are detected.

<img src="./blink_detection_with_change_point_detection.png" alt="blink detection with change point detection" width="700" height="335">

## Reference

[1] T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks," \
Proc. of Computer Vision Winter Workshop, 2016. \
[2] [Time Traveling with Data Science: Focusing on Change Point Detection in Time Series Analysis](https://www.iese.fraunhofer.de/blog/change-point-detection/)
