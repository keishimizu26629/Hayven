import 'package:flutter/services.dart';
import 'package:logger/logger.dart';

/// ネイティブコードとの連携を行うためのプラットフォームチャネル
class PlatformAudioChannel {
  // シングルトンインスタンス
  static final PlatformAudioChannel _instance =
      PlatformAudioChannel._internal();
  factory PlatformAudioChannel() => _instance;
  PlatformAudioChannel._internal();

  // ロガー
  final Logger _logger = Logger();

  // メソッドチャネル
  static const MethodChannel _methodChannel =
      MethodChannel('com.hayven/audio_method');

  // イベントチャネル
  static const EventChannel _eventChannel =
      EventChannel('com.hayven/audio_event');

  // マイク入力から音声出力への処理を開始
  Future<bool> startAudioPassthrough() async {
    try {
      final result =
          await _methodChannel.invokeMethod<bool>('startAudioPassthrough');
      return result ?? false;
    } catch (e) {
      _logger.e('マイク入力から音声出力への処理の開始に失敗しました: $e');
      return false;
    }
  }

  // マイク入力から音声出力への処理を停止
  Future<bool> stopAudioPassthrough() async {
    try {
      final result =
          await _methodChannel.invokeMethod<bool>('stopAudioPassthrough');
      return result ?? false;
    } catch (e) {
      _logger.e('マイク入力から音声出力への処理の停止に失敗しました: $e');
      return false;
    }
  }

  // 音量を設定
  Future<bool> setVolume(double volume) async {
    try {
      final result = await _methodChannel.invokeMethod<bool>(
        'setVolume',
        {'volume': volume},
      );
      return result ?? false;
    } catch (e) {
      _logger.e('音量の設定に失敗しました: $e');
      return false;
    }
  }

  // イベントストリームを取得
  Stream<dynamic> get eventStream {
    return _eventChannel.receiveBroadcastStream();
  }
}
