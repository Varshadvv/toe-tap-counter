import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class PoseKeypoint {
  final double x;
  final double y;
  final double confidence;

  const PoseKeypoint({
    required this.x,
    required this.y,
    required this.confidence,
  });
}

class PoseDetection {
  final double confidence;

  final double x;
  final double y;
  final double width;
  final double height;

  final List<PoseKeypoint> keypoints;

  const PoseDetection({
    required this.confidence,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.keypoints,
  });

  PoseKeypoint get leftAnkle => keypoints[15];

  PoseKeypoint get rightAnkle => keypoints[16];
}

class PoseDetector {
  Interpreter? _interpreter;

  int _numChannels = 56;
  int _numCandidates = 8400;

  Future<void> loadModel() async {
    final options = InterpreterOptions()..threads = 4;

    _interpreter = await Interpreter.fromAsset(
      'assets/models/yolo26n_pose.tflite',
      options: options,
    );

    final outShape = _interpreter!.getOutputTensor(0).shape;
    if (outShape.length == 3) {
      _numChannels = outShape[1];
      _numCandidates = outShape[2];
    }

    debugPrint('POSE MODEL input : ${_interpreter!.getInputTensors()}');
    debugPrint('POSE MODEL output: ${_interpreter!.getOutputTensors()}');
  }

  /// [input] is a flat Float32List (3*640*640, CHW, RGB, 0..1).
  /// Returned coordinates are normalized to the 640x640 model input.
  PoseDetection? detect(Float32List input) {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw Exception('YOLO pose model is not loaded.');
    }

    interpreter.getInputTensor(0).data = input.buffer.asUint8List();
    interpreter.invoke();

    final bytes = interpreter.getOutputTensor(0).data;
    final out = bytes.buffer.asFloat32List(
      bytes.offsetInBytes,
      bytes.lengthInBytes ~/ 4,
    );

    return _decodeBestDetection(out);
  }

  // Output layout is [1, channels, candidates], row-major:
  // value(channel c, candidate i) = out[c * candidates + i]
  PoseDetection? _decodeBestDetection(Float32List out) {
    final n = _numCandidates;

    double bestConfidence = -1;
    int bestIndex = -1;

    for (int i = 0; i < n; i++) {
      final confidence = out[4 * n + i];

      if (confidence > bestConfidence) {
        bestConfidence = confidence;
        bestIndex = i;
      }
    }

    if (bestIndex < 0) {
      return null;
    }

    final keypoints = <PoseKeypoint>[];

    for (int k = 0; k < 17; k++) {
      final base = 5 + (k * 3);

      keypoints.add(
        PoseKeypoint(
          x: out[base * n + bestIndex],
          y: out[(base + 1) * n + bestIndex],
          confidence: out[(base + 2) * n + bestIndex],
        ),
      );
    }

    return PoseDetection(
      confidence: bestConfidence,
      x: out[0 * n + bestIndex],
      y: out[1 * n + bestIndex],
      width: out[2 * n + bestIndex],
      height: out[3 * n + bestIndex],
      keypoints: keypoints,
    );
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}