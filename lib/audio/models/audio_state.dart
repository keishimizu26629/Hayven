class AudioState {
  final bool isActive;
  final bool isCryingDetected;
  final double detectionConfidence;
  final double cpuUsage;
  final double memoryUsage;

  AudioState({
    this.isActive = false,
    this.isCryingDetected = false,
    this.detectionConfidence = 0.0,
    this.cpuUsage = 0.0,
    this.memoryUsage = 0.0,
  });

  AudioState copyWith({
    bool? isActive,
    bool? isCryingDetected,
    double? detectionConfidence,
    double? cpuUsage,
    double? memoryUsage,
  }) {
    return AudioState(
      isActive: isActive ?? this.isActive,
      isCryingDetected: isCryingDetected ?? this.isCryingDetected,
      detectionConfidence: detectionConfidence ?? this.detectionConfidence,
      cpuUsage: cpuUsage ?? this.cpuUsage,
      memoryUsage: memoryUsage ?? this.memoryUsage,
    );
  }

  factory AudioState.fromMap(Map<String, dynamic> map) {
    return AudioState(
      isActive: map['isActive'] ?? false,
      isCryingDetected: map['isCryingDetected'] ?? false,
      detectionConfidence: map['detectionConfidence'] ?? 0.0,
      cpuUsage: map['cpuUsage'] ?? 0.0,
      memoryUsage: map['memoryUsage'] ?? 0.0,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'isActive': isActive,
      'isCryingDetected': isCryingDetected,
      'detectionConfidence': detectionConfidence,
      'cpuUsage': cpuUsage,
      'memoryUsage': memoryUsage,
    };
  }
}
