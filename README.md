  Toe Tap Counter

A Flutter app that counts toe taps on a football in real time. It reads the live camera feed, runs two Ultralytics YOLO models on-device (pose and ball detection), and increments a visible counter each time a foot taps the ball.

Camera -> Preprocess -> YOLO Pose(ankles) + YOLO Detect (ball) -> Tap Detection -> Counter

   Features

- Live camera preview with an overlay (green dot: left ankle, red dot: right ankle, orange box: ball)
- On-device inference with TensorFlow Lite (no network needed)
- Big `Toe Taps: N` counter that flashes on every counted tap
- Start / Reset flow (Reset stops detection, zeroes the counter and clears all tracking state)
- Tap-to-focus, and focus is locked when you press START so the picture does not hunt while counting
- Status line showing tracking state and FPS

Setup and run

 Prerequisites

- Flutter SDK (Dart `^3.13.4`, as set in `pubspec.yaml`)
- A  physical  phone (the camera and TFLite do not work well in simulators)
- Python 3 with [Ultralytics](https://docs.ultralytics.com/) installed, only needed if you have to export the models yourself (`pip install ultralytics`)

    1. Get the models

The app expects two TFLite files in `assets/models/`:

| File | Purpose |
|------|---------|
| `assets/models/yolo26n_pose.tflite` | person pose keypoints (ankles are keypoints 15 and 16) |
| `assets/models/yolo26n.tflite` | object detection (COCO class 32, "sports ball") |

If you need to export them (same commands as in py/export_model.py and py/export_ball_model.py; Ultralytics downloads the .pt weights automatically)::

python
from ultralytics import YOLO

  Pose model
YOLO("yolo26n-pose.pt").export(format="tflite", imgsz=640)

  Detection model (ball)
YOLO("yolo26n.pt").export(format="tflite", imgsz=640)


Copy the resulting `.tflite` files into `assets/models/`. Both are already declared under `assets:` in `pubspec.yaml`.

If you export at a different size (for example `imgsz=320` for more speed), change `LetterboxInfo.size` in `lib/frame_preprocessor.dart` and `_inputSize` in `lib/ball_detector.dart` to the same value.

    2. Install and run

bash
flutter pub get
flutter run --release

Use `--release`. Debug builds are many times slower for this kind of numeric Dart code and will give a very low FPS.

    3. Platform permissions

-  iOS:  add `NSCameraUsageDescription` to `ios/Runner/Info.plist`.
-  Android:  add `<uses-permission android:name="android.permission.CAMERA" />` to `android/app/src/main/AndroidManifest.xml`.

See the `camera` and `tflite_flutter` package docs for minimum OS versions.

    4. Using the app

1. Put the phone on a stand or prop it up (back camera). Stay about 2-3 m away so your full body and the ball are in frame.
2. Tap on the ball or your feet to focus. Wait until the picture is sharp.
3. Press  START . The app focuses, locks focus, and begins counting.
4. Tap the ball. The counter goes up for each tap.
5. Press  RESET  to stop and set the counter back to 0. If you move the phone, press RESET, refocus, and press START again.

The status line reads `Tracking` when both a ball and an ankle are visible, and `Show your feet and the ball` when one of them is missing.

   Project structure

lib/
  main.dart               UI, camera setup, frame loop, overlay painter
  frame_preprocessor.dart camera frame -> letterboxed 640x640 CHW float input
  pose_detector.dart      YOLO pose inference, picks best person, returns 17 keypoints
  ball_detector.dart      YOLO detection inference, returns best "sports ball" box
  tap_detector.dart       smoothing + per-foot tap state machine + counter
assets/models/
  yolo26n_pose.tflite
  yolo26n.tflite
py/
  export_model.py         exports the pose model to TFLite
  export_ball_model.py    exports the detection (ball) model to TFLite
test/
  widget_test.dart        placeholder test

   How toe taps are detected

For every processed camera frame:

1.  Preprocess.  The BGRA camera frame is letterboxed (aspect ratio preserved, gray padding) into the square model input, converted to RGB, channel-first, values 0..1. One buffer is shared by both models.
2.  Pose.  The pose model returns candidate people. The app takes the highest-confidence one (single-person assumption) and reads the left and right ankle keypoints.
3.  Ball.  The detection model's "sports ball" class gives the ball box. The highest-scoring box above a confidence of 0.25 is used. To save time, the ball model runs on every second frame, and the last ball position is reused for the skipped frames.
4.  Map back.  Model outputs are converted from letterboxed model coordinates back to normalized camera-frame coordinates for drawing and for the tap logic.
5.  Smooth.  Ankle and ball positions go through an exponential moving average. If a detection is missing, the last known position is held for 400 ms, which covers the moments when the foot hides the ball.
6.  Measure distance in ball radii.  For each foot, the distance from the ankle to the ball center is divided by the ball's radius. Using the ball's own size as the unit makes the thresholds independent of camera distance. X and Y are scaled by the frame aspect ratio so distances are not stretched.
7.  Per-foot state machine with relative hysteresis.  Each foot is tracked separately, because during toe taps one foot is almost always near the ball. A foot counts a tap when:
   - it comes toward the ball by at least `minSwing` from its recent farthest point,  and 
   - it ends up within `enterRatio` of the ball.

   After a tap the foot is "disarmed". It re-arms only after it moves away from its closest point by `minSwing` (or goes beyond `exitRatio`). A per-foot cooldown adds a second guard. This is what prevents the same tap from being counted on consecutive frames.
8.  Count.  Each tap increments the counter and flashes the label. If both feet qualify in the same frame, it still counts as one tap.

   Dependencies

| Package | Version | Used for |
|---------|---------|----------|
| `flutter` | SDK | UI framework |
| `camera` | ^0.12.1 | live preview and image stream |
| `tflite_flutter` | ^0.11.0 | running the TFLite YOLO models |
| `cupertino_icons` | ^1.0.8 | default Flutter icon font |

Dev: `flutter_test`, `flutter_lints`.

Outside Flutter:  Ultralytics  (Python) with YOLO26n and YOLO26n-pose weights, used to export the models.

   Testing

Tested by performing taps in front of the phone and comparing the on-screen count with a manual count.

| Test | Real taps | App count | Accuracy | Notes |
|------|-----------|-----------|----------|-------|
| Slow taps | 6 | 6 | 100% | Every tap counted, no double counts. |
| Fast taps | 40 | 34 | 85% | Undercounted by 6, no extra counts. At 6-10 FPS a fast tap spans only 1-2 processed frames. |

- Device used: iPhone 17
- Measured FPS:  6-10 (typically around 8), release build

During development, a tutorial toe-tap video played on a screen (7-8 real taps) was counted as 8. The same clip played at 0.25x speed counted accurately, while at 1x speed accuracy dropped. That points to frame rate, not the logic, as the main limitation (see below).

   Known limitations

-  Frame rate limits fast taps.  The app currently runs at roughly  6-10 FPS on the tested phone, because two YOLO models run on the CPU for every frame. Fast tapping (3+ taps per second) can fall between frames and be missed. Re-exporting the models at `imgsz=320`, using a GPU delegate, or running the models in background isolates are the next steps for more speed.
-  Ankle, not toe.  The pose model provides ankle keypoints only, so contact is estimated from the ankle-to-ball distance with a margin, not from the actual toe position. It is an approximation.
-  Kicks can be counted as taps.  The app counts foot-to-ball contact and does not yet check how far the ball travels afterwards, so a hard kick can count as one tap.
-  One person, one ball.  Only the highest-confidence person and ball are used. Several people or balls in view can confuse it.
-  Weak detections on non-scene content.  If the camera points at something without a person or ball, the models can still draw low-confidence ghost detections. The counter did not change in testing, but it can happen.
-  Ball hidden by the foot.  The last ball position is held for 400 ms, but longer occlusion loses the ball and taps are not counted until it reappears.
-  Camera conditions matter.  Use a fixed phone, good light, a distance of about 2-3 m, and a sharp focus. Moving the phone after START needs a RESET and a refocus.
-  Input format.  The preprocessor currently supports only the `bgra8888` camera format (the format requested in `main.dart`). Other formats (for example YUV on some Android devices) return no frame, so Android support depends on the device and may need an added conversion step. Testing was done on an iPhone.
-  Real-device only.  It is meant to run on a physical phone, in release mode.