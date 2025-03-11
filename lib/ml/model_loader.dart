import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:logger/logger.dart';

/// モデルローダークラス
///
/// TensorFlow Liteモデルをロードし、Flutterアプリで使用するためのクラス
class ModelLoader {
  // シングルトンインスタンス
  static final ModelLoader _instance = ModelLoader._internal();
  factory ModelLoader() => _instance;
  ModelLoader._internal();

  // ロガー
  final Logger _logger = Logger();

  // モデルファイルパス
  String? _modelPath;
  String? get modelPath => _modelPath;

  // 初期化フラグ
  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  /// モデルを初期化
  Future<bool> initialize() async {
    if (_isInitialized) return true;

    try {
      // モデルファイルをアセットからコピー
      final modelFile =
          await _getModelFile('assets/models/crying_detector_optimized.tflite');
      _modelPath = modelFile.path;
      _isInitialized = true;
      _logger.i('モデルを初期化しました: $_modelPath');
      return true;
    } catch (e) {
      _logger.e('モデルの初期化に失敗しました: $e');
      return false;
    }
  }

  /// アセットからモデルファイルを取得
  Future<File> _getModelFile(String assetPath) async {
    // アプリのドキュメントディレクトリを取得
    final appDir = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${appDir.path}/$fileName');

    // ファイルが存在しない場合はアセットからコピー
    if (!await file.exists()) {
      final byteData = await rootBundle.load(assetPath);
      final buffer = byteData.buffer.asUint8List();
      await file.writeAsBytes(buffer);
      _logger.i('モデルファイルをアセットからコピーしました: ${file.path}');
    }

    return file;
  }
}
