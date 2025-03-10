import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:hayven/models/audio_settings.dart';
import 'package:hayven/models/audio_state.dart';
import 'package:hayven/services/platform_audio_channel.dart';
import 'package:just_audio/just_audio.dart';
import 'package:audio_session/audio_session.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:logger/logger.dart';

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

  // 初期化
  Future<bool> initialize() async {
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

      _stateNotifier.value = AudioState();
      _logger.i('オーディオサービスが初期化されました');
      return true;
    } catch (e) {
      _logger.e('オーディオサービスの初期化に失敗しました: $e');
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

    _logger.i('オーディオ設定を更新しました: ${settings.toMap()}');
    return true;
  }

  // リソース解放
  void dispose() {
    _mockTimer?.cancel();
    _mockTimer = null;
    _eventSubscription?.cancel();
    _audioSession?.setActive(false);
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
