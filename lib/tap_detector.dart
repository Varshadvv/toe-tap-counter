import 'dart:math' as math;
import 'dart:ui' show Offset;

import 'ball_detector.dart';
import 'pose_detector.dart';

/// Smooths a noisy point with an exponential moving average and keeps the
/// last known value for [hold] when a measurement is missing.
class _Track {
  Offset? pos;
  DateTime? _seen;

  void update(Offset? m, DateTime now, double alpha, Duration hold) {
    if (m != null) {
      pos = pos == null ? m : Offset.lerp(pos!, m, alpha);
      _seen = now;
    } else if (_seen != null && now.difference(_seen!) > hold) {
      pos = null;
      _seen = null;
    }
  }

  void reset() {
    pos = null;
    _seen = null;
  }
}

/// One state machine per foot, using RELATIVE hysteresis (peak -> valley).
///
/// A tap = the foot comes toward the ball (distance falls by at least
/// [TapDetector.minSwing] radii from its recent peak) and ends up inside
/// [TapDetector.enterRatio]. After a tap, the foot must move away from its
/// lowest point by [TapDetector.minSwing] before it can tap again.
/// This works even when the feet always stay close to the ball.
class _Foot {
  final _Track track = _Track();
  double? ratio; // current distance in ball radii
  double _peak = 0;
  double _valley = 0;
  bool _armed = true;
  bool _hasPeak = false;
  DateTime? _lastTap;

  void reset() {
    track.reset();
    ratio = null;
    _peak = 0;
    _valley = 0;
    _armed = true;
    _hasPeak = false;
    _lastTap = null;
  }

  /// Returns true if this foot just made a tap.
  bool step(double? d, DateTime now) {
    ratio = d;
    if (d == null) return false;

    if (_armed) {
      if (!_hasPeak || d > _peak) {
        _peak = d;
        _hasPeak = true;
      }
      final cooled = _lastTap == null ||
          now.difference(_lastTap!) >= TapDetector.cooldown;
      if (d < TapDetector.enterRatio &&
          (_peak - d) >= TapDetector.minSwing &&
          cooled) {
        _armed = false;
        _valley = d;
        _lastTap = now;
        return true;
      }
    } else {
      if (d < _valley) _valley = d;
      // Re-arm when the foot has moved away from its closest point,
      // or is clearly far from the ball.
      if (d - _valley >= TapDetector.minSwing ||
          d > TapDetector.exitRatio) {
        _armed = true;
        _peak = d;
        _hasPeak = true;
      }
    }
    return false;
  }

  void ballLost() {
    _armed = true;
    _hasPeak = false;
  }
}

/// Counts toe taps from ankle + ball positions (frame-normalized 0..1).
class TapDetector {
  /// Frame width / height so x and y distances use the same scale.
  final double aspect;

  TapDetector({required this.aspect});

  // ---- Tunables (adjust while testing) ----
  static const double smoothing = 0.85; // weight of the NEW measurement (high = less lag at low FPS)
  static const double enterRatio = 3.0; // must end inside this (ball radii)
  static const double minSwing = 0.45; // min approach/retreat (ball radii)
  static const double exitRatio = 4.0; // far away = always re-armed
  static const double minAnkleConfidence = 0.5;
  static const Duration cooldown = Duration(milliseconds: 200); // per foot
  static const Duration hold = Duration(milliseconds: 400);

  final _Foot _left = _Foot();
  final _Foot _right = _Foot();
  final _Track _ball = _Track();
  double _ballRadius = 0.05; // in height units
  bool _hasRadius = false;

  int count = 0;

  /// Distance (in ball radii) to the closest ankle, for the status bar.
  double? lastRatio;

  void reset() {
    count = 0;
    lastRatio = null;
    _hasRadius = false;
    _left.reset();
    _right.reset();
    _ball.reset();
  }

  Offset? _toUnits(PoseKeypoint? k) {
    if (k == null || k.confidence < minAnkleConfidence) return null;
    return Offset(k.x * aspect, k.y);
  }

  /// Feed one frame. Returns true if a NEW tap was counted this frame.
  bool update({
    required PoseKeypoint? left,
    required PoseKeypoint? right,
    required BallDetection? ball,
    required DateTime now,
  }) {
    _left.track.update(_toUnits(left), now, smoothing, hold);
    _right.track.update(_toUnits(right), now, smoothing, hold);

    Offset? ballMeasurement;
    if (ball != null) {
      ballMeasurement = Offset(ball.x * aspect, ball.y);
      final r = ((ball.width * aspect) + ball.height) / 4; // mean half-size
      final clamped = math.max(r, 0.015);
      _ballRadius = _hasRadius
          ? _ballRadius + (clamped - _ballRadius) * 0.3
          : clamped;
      _hasRadius = true;
    }
    _ball.update(ballMeasurement, now, smoothing, hold);

    final b = _ball.pos;
    if (b == null) {
      _left.ballLost();
      _right.ballLost();
      lastRatio = null;
      return false;
    }

    double? ratioOf(_Foot f) {
      final a = f.track.pos;
      return a == null ? null : (a - b).distance / _ballRadius;
    }

    final dl = ratioOf(_left);
    final dr = ratioOf(_right);

    final tapL = _left.step(dl, now);
    final tapR = _right.step(dr, now);

    lastRatio = (dl == null)
        ? dr
        : (dr == null ? dl : math.min(dl, dr));

    if (tapL || tapR) {
      count++; // both feet in the same frame still counts as one tap
      return true;
    }
    return false;
  }
}