import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:hayven/audio/models/audio_settings.dart';
import 'package:hayven/audio/models/audio_state.dart';
import 'package:hayven/audio/platform_audio_channel.dart';
import 'package:just_audio/just_audio.dart';
import 'package:audio_session/audio_session.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:logger/logger.dart';
import 'package:hayven/ml/crying_detector.dart';

/// オーディオ処理サービス
///
/// 現段階ではモック実装。将来的にはネイティブコードとの連携を行う。
class AudioService {
  // シングルトンインスタンス
  static final AudioService _instance = AudioService._internal();
  factory AudioService() => _instance;
  AudioService._internal();

  // ロガー
  final Logger _logger = Logger();

  // 状態
  final ValueNotifier<AudioState> _stateNotifier = ValueNotifier(AudioState());
  ValueNotifier<AudioState> get stateNotifier => _stateNotifier;
  AudioState get state => _stateNotifier.value;

  // 設定
  AudioSettings _settings = AudioSettings();
  AudioSettings get settings => _settings;

  // オーディオセッション
  AudioSession? _audioSession;

  // プラットフォームチャネル
  final PlatformAudioChannel _platformAudioChannel = PlatformAudioChannel();

  // イベント購読
  StreamSubscription? _eventSubscription;

  // モック用タイマー（実装完了後に削除予定）
  Timer? _mockTimer;
  final Random _random = Random();

  // 泣き声検出器
  final CryingDetector _cryingDetector = CryingDetector();

  bool _isInitialized = false;
  bool _isRecording = false;
  bool _isCryingDetected = false;
  double _detectionConfidence = 0.0;

  // 音量低減レベル（0.2=20%から1.0=100%の範囲）
  double _volumeReductionLevel = 0.5;

  // 検出結果のストリームコントローラ
  final StreamController<DetectionResult> _detectionController =
      StreamController<DetectionResult>.broadcast();

  // 初期化
  Future<bool> initialize() async {
    if (_isInitialized) return true;

    _logger.i('AudioServiceを初期化中...');

    try {
      // マイク権限のリクエスト
      final status = await Permission.microphone.request();
      if (status != PermissionStatus.granted) {
        _logger.e('マイク権限が拒否されました');
        return false;
      }

      // オーディオセッションの設定
      _audioSession = await AudioSession.instance;
      await _audioSession!.configure(const AudioSessionConfiguration(
        avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
        avAudioSessionCategoryOptions:
            AVAudioSessionCategoryOptions.allowBluetooth,
        avAudioSessionMode: AVAudioSessionMode.defaultMode,
        androidAudioAttributes: AndroidAudioAttributes(
          contentType: AndroidAudioContentType.speech,
          usage: AndroidAudioUsage.voiceCommunication,
        ),
        androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
        androidWillPauseWhenDucked: true,
      ));

      // イベント購読
      _subscribeToEvents();

      // 泣き声検出器の初期化
      final detectorInitialized = await _cryingDetector.initialize();
      if (!detectorInitialized) {
        _logger.e('泣き声検出器の初期化に失敗しました');
        // 検出器の初期化に失敗してもアプリは動作可能（モック検出を使用）
      }

      _stateNotifier.value = AudioState();
      _isInitialized = true;
      _logger.i('AudioServiceの初期化が完了しました');
      return true;
    } catch (e) {
      _logger.e('AudioServiceの初期化に失敗しました: $e');
      return false;
    }
  }

  // 処理開始
  Future<bool> start() async {
    if (state.isActive) return true;

    try {
      // オーディオセッションをアクティブにする
      await _audioSession?.setActive(true);

      // マイク入力から音声出力への処理を開始
      final result = await _platformAudioChannel.startAudioPassthrough();
      if (!result) {
        _logger.e('マイク入力から音声出力への処理の開始に失敗しました');
        return false;
      }

      // 状態を更新
      _stateNotifier.value = state.copyWith(isActive: true);

      // モック実装（将来的には実際の検出ロジックに置き換え）
      _startMockDetection();

      _logger.i('オーディオ処理を開始しました');
      return true;
    } catch (e) {
      _logger.e('オーディオ処理の開始に失敗しました: $e');
      return false;
    }
  }

  // 処理停止
  Future<bool> stop() async {
    if (!state.isActive) return true;

    try {
      // マイク入力から音声出力への処理を停止
      final result = await _platformAudioChannel.stopAudioPassthrough();
      if (!result) {
        _logger.e('マイク入力から音声出力への処理の停止に失敗しました');
      }

      // オーディオセッションを非アクティブにする
      await _audioSession?.setActive(false);

      // 状態を更新
      _stateNotifier.value = state.copyWith(
        isActive: false,
        isCryingDetected: false,
        detectionConfidence: 0.0,
      );

      // モックタイマー停止
      _mockTimer?.cancel();
      _mockTimer = null;

      _logger.i('オーディオ処理を停止しました');
      return true;
    } catch (e) {
      _logger.e('オーディオ処理の停止に失敗しました: $e');
      return false;
    }
  }

  // 設定更新
  Future<bool> updateSettings(AudioSettings settings) async {
    _settings = settings;

    // 音量設定を更新
    if (state.isActive) {
      await _platformAudioChannel.setVolume(settings.volumeReduction);
    }

    // 音量低減レベルも更新
    _volumeReductionLevel = settings.volumeReduction.clamp(0.2, 1.0);

    _logger.i('オーディオ設定を更新しました: ${settings.toMap()}');
    return true;
  }

  // リソース解放
  void dispose() {
    _mockTimer?.cancel();
    _mockTimer = null;
    _eventSubscription?.cancel();
    _audioSession?.setActive(false);
    _detectionController.close();
    _logger.i('オーディオサービスのリソースを解放しました');
  }

  // イベント購読
  void _subscribeToEvents() {
    _eventSubscription?.cancel();
    _eventSubscription = _platformAudioChannel.eventStream.listen((event) {
      if (event is Map<String, dynamic>) {
        // イベントの処理
        if (event.containsKey('isCryingDetected')) {
          final isCryingDetected = event['isCryingDetected'] as bool;
          final detectionConfidence =
              event['detectionConfidence'] as double? ?? 0.0;

          _stateNotifier.value = state.copyWith(
            isCryingDetected: isCryingDetected,
            detectionConfidence: detectionConfidence,
          );
        }
      }
    }, onError: (error) {
      _logger.e('イベントストリームでエラーが発生しました: $error');
    });
  }

  // モック実装：ランダムに泣き声検出状態を変更
  void _startMockDetection() {
    _mockTimer?.cancel();
    _mockTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!state.isActive) {
        timer.cancel();
        return;
      }

      // 泣き声検出器が初期化されている場合は使用しない（実際の検出を使用）
      if (_cryingDetector.isInitialized) {
        return;
      }

      // ランダムに泣き声検出状態を変更（デモ用）
      final isCrying = _random.nextDouble() < 0.3; // 30%の確率で泣き声検出
      final confidence = isCrying ? 0.7 + _random.nextDouble() * 0.3 : 0.0;

      // 音量低減レベルに基づいて実際の低減量を計算
      final volumeReduction =
          isCrying ? confidence * _volumeReductionLevel : 0.0;

      _stateNotifier.value = state.copyWith(
        isCryingDetected: isCrying,
        detectionConfidence: confidence,
        cpuUsage: 10 + _random.nextDouble() * 5,
        memoryUsage: 50 + _random.nextDouble() * 10,
      );

      // 検出結果を通知
      _detectionController.add(DetectionResult(
        isCrying: isCrying,
        confidence: confidence,
        volumeReduction: volumeReduction,
      ));

      if (isCrying) {
        _logger.i(
            '泣き声を検出しました（信頼度: ${confidence.toStringAsFixed(2)}, 音量低減: ${(volumeReduction * 100).toStringAsFixed(0)}%）');
      }
    });
  }

  // 音量低減レベルの設定（0.2=20%から1.0=100%の範囲）
  set volumeReductionLevel(double level) {
    _volumeReductionLevel = level.clamp(0.2, 1.0);
    _logger.i('音量低減レベルを設定: $_volumeReductionLevel');

    // 設定オブジェクトも更新
    _settings = _settings.copyWith(volumeReduction: _volumeReductionLevel);
  }

  double get volumeReductionLevel => _volumeReductionLevel;

  Stream<DetectionResult> get detectionStream => _detectionController.stream;

  bool get isRecording => _isRecording;
  bool get isCryingDetected => _isCryingDetected;
  double get detectionConfidence => _detectionConfidence;

  Future<void> startRecording() async {
    if (!_isInitialized) {
      await initialize();
    }

    if (!_isRecording) {
      _logger.i('録音を開始します');
      _isRecording = true;

      // 実際のマイク入力とモデル推論の代わりに、モック実装を使用
      _startMockDetection();
    }
  }

  Future<void> stopRecording() async {
    if (_isRecording) {
      _logger.i('録音を停止します');
      _isRecording = false;
    }
  }

  // 実際の音声処理（将来的な実装）
  Future<void> _processAudio(Float32List audioData) async {
    if (!_isInitialized) return;

    try {
      // 泣き声検出
      final result =
          await _cryingDetector.detect(audioData, 16000); // サンプルレートを16kHzに設定

      _isCryingDetected = result.isCrying;
      _detectionConfidence = result.confidence;

      // 音量低減レベルに基づいて実際の低減量を計算
      final volumeReduction =
          result.isCrying ? result.confidence * _volumeReductionLevel : 0.0;

      // 検出結果を通知（音量低減情報を含む）
      _detectionController.add(DetectionResult(
        isCrying: result.isCrying,
        confidence: result.confidence,
        volumeReduction: volumeReduction,
      ));

      if (result.isCrying) {
        _logger.i(
            '泣き声を検出しました（信頼度: ${result.confidence.toStringAsFixed(2)}, 音量低減: ${(volumeReduction * 100).toStringAsFixed(0)}%）');
      }

      // 音声データの処理（音量低減）
      if (result.isCrying && volumeReduction > 0) {
        for (int i = 0; i < audioData.length; i++) {
          audioData[i] *= (1.0 - volumeReduction);
        }
      }
    } catch (e) {
      _logger.e('音声処理中にエラーが発生しました: $e');
    }
  }
}

class DetectionResult {
  final bool isCrying;
  final double confidence;
  final double volumeReduction;

  DetectionResult({
    required this.isCrying,
    required this.confidence,
    this.volumeReduction = 0.0,
  });
}
