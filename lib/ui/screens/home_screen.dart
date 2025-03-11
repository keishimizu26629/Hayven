import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:hayven/constants/app_strings.dart';
import 'package:hayven/constants/app_theme.dart';
import 'package:hayven/audio/models/audio_state.dart';
import 'package:hayven/audio/audio_provider.dart';
import 'package:hayven/ui/widgets/detection_indicator.dart';
import 'package:hayven/ui/widgets/processing_button.dart';
import 'package:provider/provider.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    // オーディオサービスの初期化
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<AudioProvider>().initialize();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppStrings.homeTitle),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              // 設定画面へ遷移（将来的に実装）
            },
          ),
        ],
      ),
      body: ValueListenableBuilder<AudioState>(
        valueListenable: context.read<AudioProvider>().stateNotifier,
        builder: (context, audioState, child) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // アプリロゴ（将来的に実装）
                const Icon(
                  Icons.hearing,
                  size: 80,
                  color: AppColors.primary,
                ),
                const SizedBox(height: 16),

                // アプリ名とタグライン
                const Text(
                  AppStrings.appName,
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: AppColors.primary,
                  ),
                ),
                const SizedBox(height: 8),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 32),
                  child: Text(
                    AppStrings.appTagline,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 16,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ),
                const SizedBox(height: 48),

                // 検出インジケーター
                DetectionIndicator(
                  isActive: audioState.isActive,
                  isCryingDetected: audioState.isCryingDetected,
                  detectionConfidence: audioState.detectionConfidence,
                ),
                const SizedBox(height: 48),

                // 処理開始/停止ボタン
                ProcessingButton(
                  isActive: audioState.isActive,
                  onPressed: () {
                    final audioProvider = context.read<AudioProvider>();
                    if (audioState.isActive) {
                      audioProvider.stop();
                    } else {
                      audioProvider.start();
                    }
                  },
                ),
                const SizedBox(height: 24),

                // 状態表示
                Text(
                  audioState.isActive
                      ? AppStrings.processingActive
                      : AppStrings.processingInactive,
                  style: TextStyle(
                    fontSize: 16,
                    color: audioState.isActive
                        ? AppColors.primary
                        : AppColors.textSecondary,
                    fontWeight: audioState.isActive
                        ? FontWeight.bold
                        : FontWeight.normal,
                  ),
                ),

                // 泣き声検出状態
                if (audioState.isActive) ...[
                  const SizedBox(height: 8),
                  Text(
                    audioState.isCryingDetected
                        ? AppStrings.cryingDetected
                        : AppStrings.noCryingDetected,
                    style: TextStyle(
                      fontSize: 14,
                      color: audioState.isCryingDetected
                          ? AppColors.error
                          : AppColors.success,
                    ),
                  )
                      .animate(
                        onPlay: (controller) => controller.repeat(),
                      )
                      .fadeIn(duration: 300.ms)
                      .then(delay: 300.ms)
                      .fadeOut(duration: 300.ms),
                ],

                // パフォーマンス情報（デバッグ用）
                if (audioState.isActive) ...[
                  const SizedBox(height: 32),
                  Text(
                    'CPU: ${audioState.cpuUsage.toStringAsFixed(1)}% | メモリ: ${audioState.memoryUsage.toStringAsFixed(1)}MB',
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.textHint,
                    ),
                  ),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
}
