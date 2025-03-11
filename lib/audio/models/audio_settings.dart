/// オーディオ設定クラス
class AudioSettings {
  /// 音量低減が有効かどうか
  final bool isVolumeReductionEnabled;

  /// 音量低減率（0.0〜1.0）
  final double volumeReduction;

  /// バックグラウンド処理が有効かどうか
  final bool isBackgroundProcessingEnabled;

  /// 高精度モードが有効かどうか
  final bool isHighPrecisionModeEnabled;

  /// 通知が有効かどうか
  final bool isNotificationEnabled;

  /// バイブレーションが有効かどうか
  final bool isVibrationEnabled;

  /// ダークモードが有効かどうか
  final bool isDarkModeEnabled;

  /// コンストラクタ
  AudioSettings({
    this.isVolumeReductionEnabled = true,
    this.volumeReduction = 0.5,
    this.isBackgroundProcessingEnabled = false,
    this.isHighPrecisionModeEnabled = false,
    this.isNotificationEnabled = true,
    this.isVibrationEnabled = true,
    this.isDarkModeEnabled = false,
  });

  /// 設定をマップに変換
  Map<String, dynamic> toMap() {
    return {
      'isVolumeReductionEnabled': isVolumeReductionEnabled,
      'volumeReduction': volumeReduction,
      'isBackgroundProcessingEnabled': isBackgroundProcessingEnabled,
      'isHighPrecisionModeEnabled': isHighPrecisionModeEnabled,
      'isNotificationEnabled': isNotificationEnabled,
      'isVibrationEnabled': isVibrationEnabled,
      'isDarkModeEnabled': isDarkModeEnabled,
    };
  }

  /// マップから設定を作成
  factory AudioSettings.fromMap(Map<String, dynamic> map) {
    return AudioSettings(
      isVolumeReductionEnabled: map['isVolumeReductionEnabled'] ?? true,
      volumeReduction: map['volumeReduction'] ?? 0.5,
      isBackgroundProcessingEnabled:
          map['isBackgroundProcessingEnabled'] ?? false,
      isHighPrecisionModeEnabled: map['isHighPrecisionModeEnabled'] ?? false,
      isNotificationEnabled: map['isNotificationEnabled'] ?? true,
      isVibrationEnabled: map['isVibrationEnabled'] ?? true,
      isDarkModeEnabled: map['isDarkModeEnabled'] ?? false,
    );
  }

  /// 設定をコピーして一部を変更
  AudioSettings copyWith({
    bool? isVolumeReductionEnabled,
    double? volumeReduction,
    bool? isBackgroundProcessingEnabled,
    bool? isHighPrecisionModeEnabled,
    bool? isNotificationEnabled,
    bool? isVibrationEnabled,
    bool? isDarkModeEnabled,
  }) {
    return AudioSettings(
      isVolumeReductionEnabled:
          isVolumeReductionEnabled ?? this.isVolumeReductionEnabled,
      volumeReduction: volumeReduction ?? this.volumeReduction,
      isBackgroundProcessingEnabled:
          isBackgroundProcessingEnabled ?? this.isBackgroundProcessingEnabled,
      isHighPrecisionModeEnabled:
          isHighPrecisionModeEnabled ?? this.isHighPrecisionModeEnabled,
      isNotificationEnabled:
          isNotificationEnabled ?? this.isNotificationEnabled,
      isVibrationEnabled: isVibrationEnabled ?? this.isVibrationEnabled,
      isDarkModeEnabled: isDarkModeEnabled ?? this.isDarkModeEnabled,
    );
  }
}
