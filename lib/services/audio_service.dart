import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:hayven/models/audio_settings.dart';
import 'package:hayven/models/audio_state.dart';

/// オーディオ処理サービス
///
/// 現段階ではモック実装。将来的にはネイティブコードとの連携を行う。
class AudioService {
  // シングルトンインスタンス
  static final AudioService _instance = AudioService._internal();
  factory AudioService() => _instance;
  AudioService._internal();

  // 状態
  final ValueNotifier<AudioState> _stateNotifier = ValueNotifier(AudioState());
  ValueNotifier<AudioState> get stateNotifier => _stateNotifier;
  AudioState get state => _stateNotifier.value;

  // 設定
  AudioSettings _settings = AudioSettings();
  AudioSettings get settings => _settings;

  // タイマー（モック実装用）
  Timer? _mockTimer;
  final Random _random = Random();

  // 初期化
  Future<bool> initialize() async {
    // 実際の実装では、ネイティブコードの初期化などを行う
    _stateNotifier.value = AudioState();
    return true;
  }

  // 処理開始
  Future<bool> start() async {
    if (state.isActive) return true;

    // 実際の実装では、ネイティブコードの処理開始を呼び出す
    _stateNotifier.value = state.copyWith(isActive: true);

    // モック実装：ランダムに泣き声検出状態を変更
    _startMockDetection();

    return true;
  }

  // 処理停止
  Future<bool> stop() async {
    if (!state.isActive) return true;

    // 実際の実装では、ネイティブコードの処理停止を呼び出す
    _stateNotifier.value = state.copyWith(
      isActive: false,
      isCryingDetected: false,
      detectionConfidence: 0.0,
    );

    // モックタイマー停止
    _mockTimer?.cancel();
    _mockTimer = null;

    return true;
  }

  // 設定更新
  Future<bool> updateSettings(AudioSettings settings) async {
    _settings = settings;

    // 実際の実装では、ネイティブコードの設定更新を呼び出す

    return true;
  }

  // リソース解放
  void dispose() {
    _mockTimer?.cancel();
    _mockTimer = null;
  }

  // モック実装：ランダムに泣き声検出状態を変更
  void _startMockDetection() {
    _mockTimer?.cancel();
    _mockTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!state.isActive) {
        timer.cancel();
        return;
      }

      // ランダムに泣き声検出状態を変更（デモ用）
      final isCrying = _random.nextDouble() < 0.3; // 30%の確率で泣き声検出
      final confidence = isCrying ? 0.7 + _random.nextDouble() * 0.3 : 0.0;

      _stateNotifier.value = state.copyWith(
        isCryingDetected: isCrying,
        detectionConfidence: confidence,
        cpuUsage: 10 + _random.nextDouble() * 5,
        memoryUsage: 50 + _random.nextDouble() * 10,
      );
    });
  }
}
