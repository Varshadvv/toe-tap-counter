import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'ball_detector.dart';
import 'frame_preprocessor.dart';
import 'pose_detector.dart';
import 'tap_detector.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final cameras = await availableCameras();

  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Toe Tap Counter',
      home: CameraScreen(cameras: cameras),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;

  final PoseDetector _poseDetector = PoseDetector();
  final BallDetector _ballDetector = BallDetector();
  final FramePreprocessor _preprocessor = FramePreprocessor();
  TapDetector? _tapDetector;

  bool _isLoadingModel = true;
  bool _isProcessingFrame = false;
  bool _isRunning = false;
  bool _isStarting = false;

  PoseKeypoint? _leftAnkle;
  PoseKeypoint? _rightAnkle;
  BallDetection? _ball;

  int _tapCount = 0;
  bool _tapFlash = false;
  int _frameIndex = 0;

  String _status = 'Loading models...';

  int _framesThisSecond = 0;
  DateTime _fpsWindowStart = DateTime.now();
  double _fps = 0;

  @override
  void initState() {
    super.initState();

    _initializeCamera();
    _loadModels();
  }

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() => _status = 'No camera found');
      return;
    }

    final camera = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => widget.cameras.first,
    );

    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.bgra8888,
    );

    try {
      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _controller = controller;
        _status = 'Camera ready';
      });

    } catch (e) {
      debugPrint('Camera initialization failed: $e');

      if (mounted) {
        setState(() => _status = 'Camera initialization failed');
      }
    }
  }

  /// Tap anywhere on the preview to focus (and meter exposure) at that point.
  Future<void> _tapToFocus(TapDownDetails d, Size size) async {
    final c = _controller;
    if (c == null || !c.value.isInitialized) return;

    final p = Offset(
      (d.localPosition.dx / size.width).clamp(0.0, 1.0),
      (d.localPosition.dy / size.height).clamp(0.0, 1.0),
    );

    try {
      await c.setFocusMode(FocusMode.auto);
      await c.setFocusPoint(p);
      await c.setExposurePoint(p);
    } catch (e) {
      debugPrint('Tap to focus not supported: $e');
    }
  }

  /// Focus on the scene, give it a moment to settle, then lock focus so it
  /// cannot hunt while counting. Exposure stays automatic.
  Future<void> _focusAndLock(CameraController c) async {
    try {
      await c.setFocusMode(FocusMode.auto);
      await Future.delayed(const Duration(milliseconds: 1200));
      await c.setFocusMode(FocusMode.locked);
    } catch (e) {
      debugPrint('Focus lock not supported: $e');
    }
  }

  Future<void> _unlockFocus(CameraController c) async {
    try {
      await c.setFocusMode(FocusMode.auto);
    } catch (e) {
      debugPrint('Focus unlock failed: $e');
    }
  }

  Future<void> _loadModels() async {
    try {
      await _poseDetector.loadModel();
      await _ballDetector.loadModel();

      if (!mounted) return;

      setState(() {
        _isLoadingModel = false;
        _status = 'Models loaded';
      });
    } catch (e) {
      debugPrint('Model loading failed: $e');

      if (!mounted) return;

      setState(() {
        _isLoadingModel = false;
        _status = 'Model failed to load';
      });
    }
  }

  Future<void> _startDetection() async {
    final controller = _controller;

    if (controller == null ||
        !controller.value.isInitialized ||
        _isLoadingModel ||
        controller.value.isStreamingImages) {
      return;
    }

    if (_isStarting) return;
    _isStarting = true;

    setState(() => _status = 'Focusing...');
    await _focusAndLock(controller);
    _isStarting = false;
    if (!mounted) return;

    // Portrait frame aspect = previewSize.height / previewSize.width
    // (previewSize is reported in landscape).
    final ps = controller.value.previewSize!;
    _tapDetector = TapDetector(aspect: ps.height / ps.width);

    _frameIndex = 0;
    _isRunning = true;

    setState(() {
      _tapCount = 0;
      _status = 'Running...';
    });

    await controller.startImageStream(_processFrame);
  }

  Future<void> _processFrame(CameraImage cameraImage) async {
    if (_isProcessingFrame || !_isRunning) return;
    _isProcessingFrame = true;

    try {
      final sw = Stopwatch()..start();
      final prepared = _preprocessor.prepare(cameraImage);
      final tPrep = sw.elapsedMilliseconds;
      if (prepared == null) return;

      final lb = prepared.letterbox;
      _frameIndex++;

      final pose = _poseDetector.detect(prepared.input);
      final tPose = sw.elapsedMilliseconds - tPrep;

      // Run the ball model every 2nd frame to save time. On skipped frames
      // the tap detector keeps the last ball position (hold window).
      final runBall = _frameIndex % 2 == 1;
      final ball = runBall ? _ballDetector.detect(prepared.input) : null;
      final tBall = sw.elapsedMilliseconds - tPrep - tPose;
      if (_frameIndex % 30 == 0) {
        debugPrint('TIMING prep ${tPrep}ms | pose ${tPose}ms | ball ${tBall}ms');
      }

      PoseKeypoint? left;
      PoseKeypoint? right;

      if (pose != null && pose.confidence >= 0.5) {
        PoseKeypoint map(PoseKeypoint k) => PoseKeypoint(
              x: lb.toFrameX(k.x),
              y: lb.toFrameY(k.y),
              confidence: k.confidence,
            );

        left = map(pose.leftAnkle);
        right = map(pose.rightAnkle);
      }

      BallDetection? mappedBall;
      if (ball != null) {
        mappedBall = BallDetection(
          confidence: ball.confidence,
          x: lb.toFrameX(ball.x),
          y: lb.toFrameY(ball.y),
          width: lb.toFrameW(ball.width),
          height: lb.toFrameH(ball.height),
        );
      }

      final tapped = _tapDetector?.update(
            left: left,
            right: right,
            ball: mappedBall,
            now: DateTime.now(),
          ) ??
          false;

      _updateFps();

      if (!mounted || !_isRunning) return;

      // lastRatio is null when the ball or both ankles are not visible.
      final tracking = _tapDetector?.lastRatio != null;

      setState(() {
        _leftAnkle = left;
        _rightAnkle = right;
        // Keep showing the last ball box on frames where we skipped the model.
        _ball = runBall ? mappedBall : _ball;
        _tapCount = _tapDetector?.count ?? _tapCount;
        if (tapped) _tapFlash = true;
        _status = (tracking ? 'Tracking' : 'Show your feet and the ball') +
            '  |  FPS ${_fps.toStringAsFixed(1)}';
      });

      if (tapped) {
        Future.delayed(const Duration(milliseconds: 150), () {
          if (mounted) setState(() => _tapFlash = false);
        });
      }
    } catch (e) {
      debugPrint('Frame processing failed: $e');
    } finally {
      _isProcessingFrame = false;
    }
  }

  void _updateFps() {
    _framesThisSecond++;

    final elapsed = DateTime.now().difference(_fpsWindowStart);
    if (elapsed.inMilliseconds >= 1000) {
      _fps = _framesThisSecond * 1000 / elapsed.inMilliseconds;
      _framesThisSecond = 0;
      _fpsWindowStart = DateTime.now();
    }
  }

  /// Reset = stop detection, zero the counter and clear all tracking state.
  Future<void> _resetDetection() async {
    final controller = _controller;
    _isRunning = false;

    if (controller != null && controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }

    _tapDetector?.reset();
    if (controller != null) _unlockFocus(controller);

    if (mounted) {
      setState(() {
        _tapCount = 0;
        _tapFlash = false;
        _status = 'Reset. Press START';
        _leftAnkle = null;
        _rightAnkle = null;
        _ball = null;
      });
    }
  }

  @override
  void dispose() {
    _isRunning = false;
    _controller?.dispose();
    _poseDetector.close();
    _ballDetector.close();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;

    if (controller == null || !controller.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final screen = MediaQuery.of(context).size;

    return Scaffold(
      body: GestureDetector(
        onTapDown: (d) => _tapToFocus(d, screen),
        child: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: controller.value.previewSize!.height,
                height: controller.value.previewSize!.width,
                child: Stack(
                  children: [
                    CameraPreview(controller),
                    Positioned.fill(
                      child: CustomPaint(
                        painter: OverlayPainter(
                          leftAnkle: _leftAnkle,
                          rightAnkle: _rightAnkle,
                          ball: _ball,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          Positioned(
            top: 50,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.65),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Toe Taps: $_tapCount',
                    style: TextStyle(
                      color: _tapFlash ? Colors.greenAccent : Colors.white,
                      fontSize: 34,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _status,
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                ],
              ),
            ),
          ),
          Positioned(
            bottom: 40,
            left: 20,
            right: 20,
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _isLoadingModel ? null : _startDetection,
                    child: const Text('START'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _resetDetection,
                    child: const Text('RESET'),
                  ),
                ),
              ],
            ),
          ),
        ],
        ),
      ),
    );
  }
}

class OverlayPainter extends CustomPainter {
  final PoseKeypoint? leftAnkle;
  final PoseKeypoint? rightAnkle;
  final BallDetection? ball;

  OverlayPainter({
    required this.leftAnkle,
    required this.rightAnkle,
    required this.ball,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final fill = Paint()..style = PaintingStyle.fill;
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    void drawAnkle(PoseKeypoint? ankle, Color color) {
      if (ankle == null || ankle.confidence < 0.5) return;

      final center = Offset(ankle.x * size.width, ankle.y * size.height);

      fill.color = color;
      canvas.drawCircle(center, 12, fill);

      stroke.color = Colors.white;
      canvas.drawCircle(center, 12, stroke);
    }

    drawAnkle(leftAnkle, Colors.green);
    drawAnkle(rightAnkle, Colors.red);

    final b = ball;
    if (b != null) {
      final rect = Rect.fromCenter(
        center: Offset(b.x * size.width, b.y * size.height),
        width: b.width * size.width,
        height: b.height * size.height,
      );

      stroke.color = Colors.orange;
      canvas.drawRect(rect, stroke);
    }
  }

  @override
  bool shouldRepaint(covariant OverlayPainter oldDelegate) => true;
}