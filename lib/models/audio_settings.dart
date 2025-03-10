class AudioSettings {
  final double detectionThreshold;
  final double volumeReduction;
  final bool enableBackgroundMode;

  AudioSettings({
    this.detectionThreshold = 0.7,
    this.volumeReduction = -20.0,
    this.enableBackgroundMode = true,
  });

  AudioSettings copyWith({
    double? detectionThreshold,
    double? volumeReduction,
    bool? enableBackgroundMode,
  }) {
    return AudioSettings(
      detectionThreshold: detectionThreshold ?? this.detectionThreshold,
      volumeReduction: volumeReduction ?? this.volumeReduction,
      enableBackgroundMode: enableBackgroundMode ?? this.enableBackgroundMode,
    );
  }

  factory AudioSettings.fromMap(Map<String, dynamic> map) {
    return AudioSettings(
      detectionThreshold: map['detectionThreshold'] ?? 0.7,
      volumeReduction: map['volumeReduction'] ?? -20.0,
      enableBackgroundMode: map['enableBackgroundMode'] ?? true,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'detectionThreshold': detectionThreshold,
      'volumeReduction': volumeReduction,
      'enableBackgroundMode': enableBackgroundMode,
    };
  }
}
