import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class BallDetection {
  final double confidence;

  /// Center and size, normalized to 0..1.
  final double x;
  final double y;
  final double width;
  final double height;

  const BallDetection({
    required this.confidence,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
}

class BallDetector {
  static const int _inputSize = 640;
  static const int _sportsBallClassId = 32; // COCO "sports ball"
  static const double _minConfidence = 0.25; // balls are small; keep low

  Interpreter? _interpreter;
  int _numChannels = 84;
  int _numCandidates = 8400;

  Future<void> loadModel() async {
    final options = InterpreterOptions()..threads = 4;

    _interpreter = await Interpreter.fromAsset(
      'assets/models/yolo26n.tflite',
      options: options,
    );

    final outShape = _interpreter!.getOutputTensor(0).shape;
    debugPrint('BALL MODEL input : ${_interpreter!.getInputTensors()}');
    debugPrint('BALL MODEL output: ${_interpreter!.getOutputTensors()}');

    if (outShape.length == 3) {
      _numChannels = outShape[1];
      _numCandidates = outShape[2];
    }
  }

  /// [input] is a flat Float32List of length 3*640*640 in CHW order,
  /// RGB, values 0..1 (the same buffer you feed the pose model).
  BallDetection? detect(Float32List input) {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw Exception('Ball model is not loaded.');
    }

    interpreter.getInputTensor(0).data = input.buffer.asUint8List();
    interpreter.invoke();

    final bytes = interpreter.getOutputTensor(0).data;
    final out = bytes.buffer.asFloat32List(
      bytes.offsetInBytes,
      bytes.lengthInBytes ~/ 4,
    );

    return _decodeBestBall(out);
  }

  // Layout [1, channels, candidates]: value(c, i) = out[c * candidates + i]
  BallDetection? _decodeBestBall(Float32List out) {
    final n = _numCandidates;
    final classRow = 4 + _sportsBallClassId;

    if (classRow >= _numChannels) {
      debugPrint('Unexpected output shape, no class row $classRow');
      return null;
    }

    double bestScore = _minConfidence;
    int bestIndex = -1;

    for (int i = 0; i < n; i++) {
      final score = out[classRow * n + i];
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }

    if (bestIndex < 0) {
      return null;
    }

    double cx = out[0 * n + bestIndex];
    double cy = out[1 * n + bestIndex];
    double w = out[2 * n + bestIndex];
    double h = out[3 * n + bestIndex];

    // Some exports give pixel values (0..640), others normalized (0..1).
    if (cx > 1.5 || cy > 1.5 || w > 1.5 || h > 1.5) {
      cx /= _inputSize;
      cy /= _inputSize;
      w /= _inputSize;
      h /= _inputSize;
    }

    return BallDetection(
      confidence: bestScore,
      x: cx,
      y: cy,
      width: w,
      height: h,
    );
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}