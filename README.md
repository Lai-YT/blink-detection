# Blink Detection

This study supports the *blink* module in [Lai-YT/webcam-applications](https://github.com/Lai-YT/webcam-applications).

## Purpose

We want to detect whether the user blinks or not from the video.

### Implemention

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

## How to start?

The detection works under the `main.py` file, 2 modes are provided, one is the \
video mode, the other is the live stream mode, which opens the webcam.

```
python main.py [./$(file_path) | live]
```

### video mode

By specifying a video file path, you use the video mode. You are suggested to put \
the video under `video/`. \
The whole video will be blink detected, which takes approximately as long \
as the video time, and generates a *txt* file which has the same name as your \
video file to the `video/` folder, that is the *EAR*s of all the frames in the video.

### live stream mode

By providing `live` as command line argument, you use the live stream mode. \
The webcam will be opened and captures frames of the user. Blink detection works \
in real time and shows the current blink count onto the window. After closing \
the live stream with `q`, a *txt* named `ratio-%Y%m%d-%H%M%S` is generated next \
to the main file, that is the is the *EAR*s of all the frames in the live stream.

### Plot EARs and blinks

```
python plot_ratios.py ./$(file_path) [show]
```

The file arguments is the *txt* *EAR* file generated by `main.py`. If `show` \
argument is provided, the plot shows directly without storing as a image file; \
if not provided, the *png* file is stored to the `plots/` folder with the same \
name as the *EAR* file.

### Plot the real blinks

If you want to compare the real blinks with the detected blinks, you can use the \
annotation file.

```
python annotate_blink.py $(video_to_annotate)
```

The frames of the video will be showed one by one, if a blink occurs, key in \
`o`, otherwise `x`. Notice that a single blink may take more than 1 frame, but \
only mark one of them as `o`. \
After the annotation, a *json* file named `ann-%Y%m%d-%H%M%S` is generated next \
to the *annotate_blink.py*, this is because you may annotate a single video \
several times. \
Choose the one you think best fits the real blinks, rename it with the same name \
as the *EAR* file (keep the suffix *json*) and put it under `video/`. \
Then re-run `plot_ratios.py`, you'll see the real-blinks are plotted with red dots!

## Reference

[1] T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks," \
Proc. of Computer Vision Winter Workshop, 2016. \
[2] [Time Traveling with Data Science: Focusing on Change Point Detection in Time Series Analysis](https://www.iese.fraunhofer.de/blog/change-point-detection/)

## License

Distributed under [MIT License](./LICENSE).
