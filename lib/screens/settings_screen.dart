import 'package:flutter/material.dart';
import 'package:hayven/audio/audio_service.dart';
import 'package:hayven/audio/models/audio_settings.dart';
import 'package:hayven/theme/app_theme.dart';
import 'package:hayven/widgets/app_bar.dart';
import 'package:hayven/widgets/settings_card.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({Key? key}) : super(key: key);

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final AudioService _audioService = AudioService();
  late AudioSettings _settings;

  // 音量低減レベル（0.2=20%から1.0=100%の範囲）
  late double _volumeReductionLevel;

  @override
  void initState() {
    super.initState();
    _settings = _audioService.settings;
    _volumeReductionLevel = _audioService.volumeReductionLevel;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const HayvenAppBar(title: '設定'),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildAudioSettings(),
            const SizedBox(height: 24),
            _buildNotificationSettings(),
            const SizedBox(height: 24),
            _buildAppearanceSettings(),
          ],
        ),
      ),
    );
  }

  Widget _buildAudioSettings() {
    return SettingsCard(
      title: '音声設定',
      children: [
        // 泣き声検出時の音量低減
        ListTile(
          title: const Text('泣き声検出時の音量低減'),
          subtitle: Text('${(_settings.volumeReduction * 100).toInt()}%'),
          trailing: Switch(
            value: _settings.isVolumeReductionEnabled,
            onChanged: (value) {
              setState(() {
                _settings = _settings.copyWith(isVolumeReductionEnabled: value);
                _audioService.updateSettings(_settings);
              });
            },
          ),
        ),

        // 音量低減レベルのスライダー
        if (_settings.isVolumeReductionEnabled)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('音量低減レベル', style: TextStyle(fontSize: 14)),
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Text('20%', style: TextStyle(fontSize: 12)),
                    Expanded(
                      child: Slider(
                        value: _volumeReductionLevel,
                        min: 0.2,
                        max: 1.0,
                        divisions: 8,
                        label: '${(_volumeReductionLevel * 100).toInt()}%',
                        onChanged: (value) {
                          setState(() {
                            _volumeReductionLevel = value;
                          });
                        },
                        onChangeEnd: (value) {
                          _audioService.volumeReductionLevel = value;
                        },
                      ),
                    ),
                    const Text('100%', style: TextStyle(fontSize: 12)),
                  ],
                ),
                Center(
                  child: Text(
                    _volumeReductionLevel >= 0.9
                        ? '完全に除去'
                        : '${(_volumeReductionLevel * 100).toInt()}%低減',
                    style: TextStyle(
                      fontSize: 12,
                      color: AppTheme.primaryColor,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),

        // バックグラウンド処理
        ListTile(
          title: const Text('バックグラウンド処理'),
          subtitle: const Text('アプリがバックグラウンドでも処理を継続'),
          trailing: Switch(
            value: _settings.isBackgroundProcessingEnabled,
            onChanged: (value) {
              setState(() {
                _settings =
                    _settings.copyWith(isBackgroundProcessingEnabled: value);
                _audioService.updateSettings(_settings);
              });
            },
          ),
        ),

        // 高精度モード
        ListTile(
          title: const Text('高精度モード'),
          subtitle: const Text('より正確な検出（バッテリー消費増加）'),
          trailing: Switch(
            value: _settings.isHighPrecisionModeEnabled,
            onChanged: (value) {
              setState(() {
                _settings =
                    _settings.copyWith(isHighPrecisionModeEnabled: value);
                _audioService.updateSettings(_settings);
              });
            },
          ),
        ),
      ],
    );
  }

  Widget _buildNotificationSettings() {
    return SettingsCard(
      title: '通知設定',
      children: [
        ListTile(
          title: const Text('泣き声検出通知'),
          subtitle: const Text('泣き声を検出したときに通知'),
          trailing: Switch(
            value: _settings.isNotificationEnabled,
            onChanged: (value) {
              setState(() {
                _settings = _settings.copyWith(isNotificationEnabled: value);
                _audioService.updateSettings(_settings);
              });
            },
          ),
        ),
        ListTile(
          title: const Text('バイブレーション'),
          subtitle: const Text('泣き声検出時に振動'),
          trailing: Switch(
            value: _settings.isVibrationEnabled,
            onChanged: (value) {
              setState(() {
                _settings = _settings.copyWith(isVibrationEnabled: value);
                _audioService.updateSettings(_settings);
              });
            },
          ),
        ),
      ],
    );
  }

  Widget _buildAppearanceSettings() {
    return SettingsCard(
      title: '外観設定',
      children: [
        ListTile(
          title: const Text('ダークモード'),
          subtitle: const Text('ダークテーマを使用'),
          trailing: Switch(
            value: _settings.isDarkModeEnabled,
            onChanged: (value) {
              setState(() {
                _settings = _settings.copyWith(isDarkModeEnabled: value);
                _audioService.updateSettings(_settings);
              });
            },
          ),
        ),
      ],
    );
  }
}
