import 'package:flutter/material.dart';
import 'package:hayven/constants/app_strings.dart';
import 'package:hayven/constants/app_theme.dart';

class ProcessingButton extends StatelessWidget {
  final bool isActive;
  final VoidCallback onPressed;

  const ProcessingButton({
    super.key,
    required this.isActive,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: isActive ? AppColors.error : AppColors.primary,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30),
        ),
        elevation: 5,
        shadowColor: isActive
            ? AppColors.error.withOpacity(0.5)
            : AppColors.primary.withOpacity(0.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            isActive ? Icons.stop : Icons.play_arrow,
            size: 28,
          ),
          const SizedBox(width: 8),
          Text(
            isActive ? AppStrings.stopProcessing : AppStrings.startProcessing,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
