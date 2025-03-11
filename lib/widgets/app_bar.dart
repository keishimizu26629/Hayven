import 'package:flutter/material.dart';

/// アプリケーション共通のアプリバー
class HayvenAppBar extends StatelessWidget implements PreferredSizeWidget {
  /// タイトル
  final String title;

  /// アクションボタン
  final List<Widget>? actions;

  /// 戻るボタンを表示するかどうか
  final bool showBackButton;

  /// 戻るボタンを押したときの処理
  final VoidCallback? onBackPressed;

  /// コンストラクタ
  const HayvenAppBar({
    Key? key,
    required this.title,
    this.actions,
    this.showBackButton = true,
    this.onBackPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      title: Text(
        title,
        style: const TextStyle(
          fontWeight: FontWeight.bold,
        ),
      ),
      centerTitle: true,
      elevation: 0,
      leading: showBackButton
          ? IconButton(
              icon: const Icon(Icons.arrow_back_ios),
              onPressed: onBackPressed ?? () => Navigator.of(context).pop(),
            )
          : null,
      actions: actions,
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
