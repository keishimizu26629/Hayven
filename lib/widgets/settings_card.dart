import 'package:flutter/material.dart';

/// 設定画面で使用するカードウィジェット
class SettingsCard extends StatelessWidget {
  /// カードのタイトル
  final String title;

  /// カードの子ウィジェット
  final List<Widget> children;

  /// カードの余白
  final EdgeInsetsGeometry padding;

  /// コンストラクタ
  const SettingsCard({
    Key? key,
    required this.title,
    required this.children,
    this.padding = const EdgeInsets.all(8.0),
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: padding,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const Divider(),
            ...children,
          ],
        ),
      ),
    );
  }
}
