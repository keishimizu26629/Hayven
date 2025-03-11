import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:hayven/constants/app_theme.dart';

class DetectionIndicator extends StatelessWidget {
  final bool isActive;
  final bool isCryingDetected;
  final double detectionConfidence;

  const DetectionIndicator({
    super.key,
    required this.isActive,
    required this.isCryingDetected,
    required this.detectionConfidence,
  });

  @override
  Widget build(BuildContext context) {
    if (!isActive) {
      return _buildInactiveIndicator();
    }

    return _buildActiveIndicator();
  }

  Widget _buildInactiveIndicator() {
    return Container(
      width: 200,
      height: 200,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: AppColors.surfaceDark,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            spreadRadius: 2,
          ),
        ],
      ),
      child: const Center(
        child: Icon(
          Icons.mic_off,
          size: 60,
          color: AppColors.textHint,
        ),
      ),
    );
  }

  Widget _buildActiveIndicator() {
    final Color indicatorColor =
        isCryingDetected ? AppColors.error : AppColors.success;

    return Stack(
      alignment: Alignment.center,
      children: [
        // 外側の円（パルスアニメーション）
        if (isCryingDetected)
          Container(
            width: 220,
            height: 220,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: indicatorColor.withOpacity(0.2),
            ),
          )
              .animate(
                onPlay: (controller) => controller.repeat(),
              )
              .scale(
                begin: const Offset(0.8, 0.8),
                end: const Offset(1.2, 1.2),
                duration: 1000.ms,
              )
              .then(delay: 200.ms)
              .fadeOut(duration: 800.ms),

        // 中間の円
        Container(
          width: 200,
          height: 200,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: AppColors.surfaceLight,
            border: Border.all(
              color: indicatorColor,
              width: 4,
            ),
            boxShadow: [
              BoxShadow(
                color: indicatorColor.withOpacity(0.3),
                blurRadius: 10,
                spreadRadius: 2,
              ),
            ],
          ),
        ),

        // 内側の円（検出信頼度に応じたサイズ）
        Container(
          width: 160 * (isCryingDetected ? detectionConfidence : 0.3),
          height: 160 * (isCryingDetected ? detectionConfidence : 0.3),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: indicatorColor.withOpacity(0.3),
          ),
        ).animate().scale(
              duration: 300.ms,
              curve: Curves.easeOutBack,
            ),

        // アイコン
        Icon(
          isCryingDetected ? Icons.child_care : Icons.mic,
          size: 60,
          color: indicatorColor,
        ),
      ],
    );
  }
}
