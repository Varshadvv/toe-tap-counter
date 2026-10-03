import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

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

  int sensorOrientation = 90;

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

  PreparedFrame? prepare(CameraImage image) {
    switch (image.format.group) {
      case ImageFormatGroup.bgra8888:
        return _prepareBgra(image);
      case ImageFormatGroup.yuv420:
        return _prepareYuv(image);
      default:
        return null;
    }
  }

  // ===========================================================================
  // BGRA (iOS). Unchanged logic.
  // ===========================================================================

  int _cw = -1;
  int _ch = -1;
  int _cStride = -1;
  int _newW = 0;
  int _newH = 0;
  int _offX = 0;
  int _offY = 0;
  Int32List _srcXByte = Int32List(0); // dst x -> byte offset in a source row
  Int32List _srcRowStart = Int32List(0); // dst y -> byte offset of source row
  LetterboxInfo? _info;

  void _rebuildBgraCache(int w, int h, int stride) {
    final scale = size / (w > h ? w : h);
    _newW = (w * scale).round();
    _newH = (h * scale).round();
    _offX = (size - _newW) ~/ 2;
    _offY = (size - _newH) ~/ 2;

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
      padY: _offY.toDouble(),
    );

    // Invalidate the YUV cache: the shared buffer padding was just rewritten.
    _yw = -1;

    _cw = w;
    _ch = h;
    _cStride = stride;
  }

  PreparedFrame? _prepareBgra(CameraImage image) {
    final plane = image.planes.first;
    final bytes = plane.bytes;
    final stride = plane.bytesPerRow;
    final w = image.width;
    final h = image.height;

    if (w != _cw || h != _ch || stride != _cStride) {
      _rebuildBgraCache(w, h, stride);
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

    return PreparedFrame(buf, _info!);
  }

  // ===========================================================================
  // YUV420 (Android). Converts to RGB and rotates to upright in one pass.
  // ===========================================================================

  int _yw = -1;
  int _yh = -1;
  int _yRot = -1;
  int _yNewW = 0;
  int _yNewH = 0;
  int _yOffX = 0;
  int _yOffY = 0;

  // Source coordinate = (column part) + (row part), see _rebuildYuvCache.
  Int32List _cxSx = Int32List(0);
  Int32List _cxSy = Int32List(0);
  Int32List _rySx = Int32List(0);
  Int32List _rySy = Int32List(0);
  LetterboxInfo? _yuvInfo;
  bool _loggedYuv = false;

  /// Clockwise rotation (degrees) needed to make the raw frame upright.
  int _rotationFor(int w, int h) {
    if (w <= h) return 0; // already portrait, do not rotate
    return sensorOrientation == 270 ? 270 : 90;
  }

  void _rebuildYuvCache(int w, int h, int rot) {
    // Size of the upright frame.
    final uw = rot == 0 ? w : h;
    final uh = rot == 0 ? h : w;

    final scale = size / (uw > uh ? uw : uh);
    _yNewW = (uw * scale).round();
    _yNewH = (uh * scale).round();
    _yOffX = (size - _yNewW) ~/ 2;
    _yOffY = (size - _yNewH) ~/ 2;

    // For an upright pixel (ux, uy), the source pixel (sx, sy) is:
    //   rot 0  : sx = ux          , sy = uy
    //   rot 90 : sx = uy          , sy = h - 1 - ux   (clockwise)
    //   rot 270: sx = w - 1 - uy  , sy = ux           (counter-clockwise)
    // Each of sx and sy depends on only one of (ux, uy), so we precompute
    // a per-column array and a per-row array and add them.
    _cxSx = Int32List(_yNewW);
    _cxSy = Int32List(_yNewW);
    for (int x = 0; x < _yNewW; x++) {
      final ux = (x / scale).floor().clamp(0, uw - 1).toInt();
      if (rot == 0) {
        _cxSx[x] = ux;
      } else if (rot == 90) {
        _cxSy[x] = h - 1 - ux;
      } else {
        _cxSy[x] = ux;
      }
    }

    _rySx = Int32List(_yNewH);
    _rySy = Int32List(_yNewH);
    for (int y = 0; y < _yNewH; y++) {
      final uy = (y / scale).floor().clamp(0, uh - 1).toInt();
      if (rot == 0) {
        _rySy[y] = uy;
      } else if (rot == 90) {
        _rySx[y] = uy;
      } else {
        _rySx[y] = w - 1 - uy;
      }
    }

    _buffer.fillRange(0, _buffer.length, 114 / 255);

    _yuvInfo = LetterboxInfo(
      frameWidth: uw,
      frameHeight: uh,
      scale: scale,
      padX: _yOffX.toDouble(),
      padY: _yOffY.toDouble(),
    );

    // Invalidate the BGRA cache: the shared buffer padding was just rewritten.
    _cw = -1;

    _yw = w;
    _yh = h;
    _yRot = rot;
  }

  PreparedFrame? _prepareYuv(CameraImage image) {
    if (image.planes.length < 3) {
      debugPrint('YUV frame has ${image.planes.length} planes, expected 3');
      return null;
    }

    final w = image.width;
    final h = image.height;
    final rot = _rotationFor(w, h);

    if (w != _yw || h != _yh || rot != _yRot) {
      _rebuildYuvCache(w, h, rot);
    }

    final yP = image.planes[0];
    final uP = image.planes[1];
    final vP = image.planes[2];
    final yBytes = yP.bytes;
    final uBytes = uP.bytes;
    final vBytes = vP.bytes;
    final yStride = yP.bytesPerRow;
    final uStride = uP.bytesPerRow;
    final vStride = vP.bytesPerRow;
    final uPix = uP.bytesPerPixel ?? 1;
    final vPix = vP.bytesPerPixel ?? 1;

    if (!_loggedYuv) {
      _loggedYuv = true;
      debugPrint('YUV frame ${w}x$h rot=$rot sensor=$sensorOrientation '
          'yStride=$yStride uStride=$uStride uPix=$uPix '
          'vStride=$vStride vPix=$vPix');
    }

    final buf = _buffer;
    final lut = _lut;
    final cxSx = _cxSx;
    final cxSy = _cxSy;
    final rySx = _rySx;
    final rySy = _rySy;
    final newW = _yNewW;
    final newH = _yNewH;
    final offX = _yOffX;
    final offY = _yOffY;

    for (int y = 0; y < newH; y++) {
      final rx = rySx[y];
      final ry = rySy[y];
      int d = (y + offY) * size + offX;

      for (int x = 0; x < newW; x++, d++) {
        final sx = cxSx[x] + rx;
        final sy = cxSy[x] + ry;

        final yy = yBytes[sy * yStride + sx];
        final cx = sx >> 1;
        final cy = sy >> 1;
        final u = uBytes[cy * uStride + cx * uPix] - 128;
        final v = vBytes[cy * vStride + cx * vPix] - 128;

        int r = yy + ((359 * v) >> 8); // 1.402 * v
        int g = yy - ((88 * u + 183 * v) >> 8); // 0.344 * u + 0.714 * v
        int b = yy + ((454 * u) >> 8); // 1.772 * u

        r = r < 0 ? 0 : (r > 255 ? 255 : r);
        g = g < 0 ? 0 : (g > 255 ? 255 : g);
        b = b < 0 ? 0 : (b > 255 ? 255 : b);

        buf[d] = lut[r];
        buf[_area + d] = lut[g];
        buf[2 * _area + d] = lut[b];
      }
    }

    return PreparedFrame(buf, _yuvInfo!);
  }
}