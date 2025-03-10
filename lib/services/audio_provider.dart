import 'package:flutter/foundation.dart';
import 'package:hayven/models/audio_settings.dart';
import 'package:hayven/models/audio_state.dart';
import 'package:hayven/services/audio_service.dart';

class AudioProvider extends ChangeNotifier {
  final AudioService _audioService = AudioService();

  // 状態
  AudioState get state => _audioService.state;
  ValueNotifier<AudioState> get stateNotifier => _audioService.stateNotifier;

  // 設定
  AudioSettings get settings => _audioService.settings;

  // 初期化
  Future<bool> initialize() async {
    final result = await _audioService.initialize();
    notifyListeners();
    return result;
  }

  // 処理開始
  Future<bool> start() async {
    final result = await _audioService.start();
    notifyListeners();
    return result;
  }

  // 処理停止
  Future<bool> stop() async {
    final result = await _audioService.stop();
    notifyListeners();
    return result;
  }

  // 設定更新
  Future<bool> updateSettings(AudioSettings settings) async {
    final result = await _audioService.updateSettings(settings);
    notifyListeners();
    return result;
  }

  @override
  void dispose() {
    _audioService.dispose();
    super.dispose();
  }
}
