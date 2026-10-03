import 'dart:typed_data';

import 'package:camera/camera.dart';

/// Describes how a camera frame was padded into the square model input,
/// so model outputs can be mapped back to normalized frame coordinates.
class LetterboxInfo {
  /// Model input size. Change this ONE value (e.g. 320) after re-exporting
  /// the models with the same imgsz.
  static const int size = 640;

  final int frameWidth;
  final int frameHeight;
  final double scale;
  final double padX;
  final double padY;

  const LetterboxInfo({
    required this.frameWidth,
    required this.frameHeight,
    required this.scale,
    required this.padX,
    required this.padY,
  });

  /// Model-normalized (0..1 of the square) -> frame-normalized (0..1).
  double toFrameX(double nx) => ((nx * size) - padX) / scale / frameWidth;
  double toFrameY(double ny) => ((ny * size) - padY) / scale / frameHeight;
  double toFrameW(double nw) => nw * size / scale / frameWidth;
  double toFrameH(double nh) => nh * size / scale / frameHeight;
}

class PreparedFrame {
  final Float32List input; // CHW, RGB, 0..1, length 3*size*size
  final LetterboxInfo letterbox;

  const PreparedFrame(this.input, this.letterbox);
}

class FramePreprocessor {
  static const int size = LetterboxInfo.size;
  static const int _area = size * size;

  // Reused every frame (the models only read from it).
  final Float32List _buffer = Float32List(3 * _area);

  // byte value -> 0..1 float, avoids a division per channel per pixel.
  static final Float32List _lut = _buildLut();
  static Float32List _buildLut() {
    final l = Float32List(256);
    for (int i = 0; i < 256; i++) {
      l[i] = i / 255.0;
    }
    return l;
  }

  // ---- Cache, rebuilt only when the frame geometry changes ----
  int _cw = -1;
  int _ch = -1;
  int _cStride = -1;
  int _newW = 0;
  int _newH = 0;
  int _offX = 0;
  late Int32List _srcXByte; // dst x -> byte offset of source pixel in a row
  late Int32List _srcRowStart; // dst y -> byte offset of source row
  late LetterboxInfo _info;

  void _rebuildCache(int w, int h, int stride) {
    final scale = size / (w > h ? w : h);
    _newW = (w * scale).round();
    _newH = (h * scale).round();
    _offX = (size - _newW) ~/ 2;
    final offY = (size - _newH) ~/ 2;

    _srcXByte = Int32List(_newW);
    for (int x = 0; x < _newW; x++) {
      final srcX = (x / scale).floor().clamp(0, w - 1).toInt();
      _srcXByte[x] = srcX * 4;
    }

    _srcRowStart = Int32List(_newH);
    for (int y = 0; y < _newH; y++) {
      final srcY = (y / scale).floor().clamp(0, h - 1).toInt();
      _srcRowStart[y] = srcY * stride;
    }

    // Gray padding (value Ultralytics uses). Padding is never overwritten
    // afterwards, so filling once per geometry change is enough.
    _buffer.fillRange(0, _buffer.length, 114 / 255);

    _info = LetterboxInfo(
      frameWidth: w,
      frameHeight: h,
      scale: scale,
      padX: _offX.toDouble(),
      padY: offY.toDouble(),
    );

    _offY = offY;
    _cw = w;
    _ch = h;
    _cStride = stride;
  }

  int _offY = 0;

  PreparedFrame? prepare(CameraImage image) {
    if (image.format.group != ImageFormatGroup.bgra8888) {
      return null;
    }

    final plane = image.planes.first;
    final bytes = plane.bytes;
    final stride = plane.bytesPerRow;
    final w = image.width;
    final h = image.height;

    if (w != _cw || h != _ch || stride != _cStride) {
      _rebuildCache(w, h, stride);
    }

    final buf = _buffer;
    final lut = _lut;
    final srcXByte = _srcXByte;
    final srcRowStart = _srcRowStart;
    final newW = _newW;
    final newH = _newH;
    final offX = _offX;
    final offY = _offY;

    for (int y = 0; y < newH; y++) {
      final rowStart = srcRowStart[y];
      int d = (y + offY) * size + offX;

      for (int x = 0; x < newW; x++, d++) {
        final i = rowStart + srcXByte[x]; // B, G, R, A
        buf[d] = lut[bytes[i + 2]]; // R
        buf[_area + d] = lut[bytes[i + 1]]; // G
        buf[2 * _area + d] = lut[bytes[i]]; // B
      }
    }

    return PreparedFrame(buf, _info);
  }
}