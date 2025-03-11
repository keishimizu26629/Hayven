import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:logger/logger.dart';
import 'package:hayven/ml/model_loader.dart';

/// 泣き声検出クラス
///
/// TensorFlow Liteモデルを使用して泣き声を検出するクラス
class CryingDetector {
  // シングルトンインスタンス
  static final CryingDetector _instance = CryingDetector._internal();
  factory CryingDetector() => _instance;
  CryingDetector._internal();

  // ロガー
  final Logger _logger = Logger();

  // モデルローダー
  final ModelLoader _modelLoader = ModelLoader();

  // メソッドチャネル
  static const MethodChannel _methodChannel =
      MethodChannel('com.hayven/ml_method');

  // 初期化フラグ
  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  /// 初期化
  Future<bool> initialize() async {
    if (_isInitialized) return true;

    try {
      // モデルローダーを初期化
      final modelInitialized = await _modelLoader.initialize();
      if (!modelInitialized) {
        _logger.e('モデルローダーの初期化に失敗しました');
        return false;
      }

      // ネイティブコードでモデルを初期化
      final result = await _methodChannel.invokeMethod<bool>(
        'initializeModel',
        {'modelPath': _modelLoader.modelPath},
      );

      _isInitialized = result ?? false;
      _logger.i('泣き声検出器を初期化しました: $_isInitialized');
      return _isInitialized;
    } catch (e) {
      _logger.e('泣き声検出器の初期化に失敗しました: $e');
      return false;
    }
  }

  /// 音声データから泣き声を検出
  Future<DetectionResult> detect(Float32List audioData, int sampleRate) async {
    if (!_isInitialized) {
      _logger.e('泣き声検出器が初期化されていません');
      return DetectionResult(isCrying: false, confidence: 0.0);
    }

    try {
      // ネイティブコードで推論を実行
      final result = await _methodChannel.invokeMethod<Map<dynamic, dynamic>>(
        'detectCrying',
        {
          'audioData': audioData,
          'sampleRate': sampleRate,
        },
      );

      if (result == null) {
        _logger.e('推論結果がnullです');
        return DetectionResult(isCrying: false, confidence: 0.0);
      }

      final isCrying = result['isCrying'] as bool;
      final confidence = result['confidence'] as double;

      _logger.i('泣き声検出結果: isCrying=$isCrying, confidence=$confidence');
      return DetectionResult(isCrying: isCrying, confidence: confidence);
    } catch (e) {
      _logger.e('泣き声検出中にエラーが発生しました: $e');
      return DetectionResult(isCrying: false, confidence: 0.0);
    }
  }
}

/// 検出結果クラス
class DetectionResult {
  final bool isCrying;
  final double confidence;

  DetectionResult({
    required this.isCrying,
    required this.confidence,
  });
}
