import 'package:flutter/material.dart';

/// アプリのテーマ定義
class AppTheme {
  /// プライマリカラー
  static const Color primaryColor = Color(0xFF6200EE);

  /// セカンダリカラー
  static const Color secondaryColor = Color(0xFF03DAC6);

  /// 背景色（ライトモード）
  static const Color backgroundColorLight = Color(0xFFF5F5F5);

  /// 背景色（ダークモード）
  static const Color backgroundColorDark = Color(0xFF121212);

  /// カードの背景色（ライトモード）
  static const Color cardColorLight = Colors.white;

  /// カードの背景色（ダークモード）
  static const Color cardColorDark = Color(0xFF1E1E1E);

  /// テキスト色（ライトモード）
  static const Color textColorLight = Color(0xFF333333);

  /// テキスト色（ダークモード）
  static const Color textColorDark = Color(0xFFEEEEEE);

  /// ライトモードのテーマ
  static ThemeData lightTheme = ThemeData(
    primaryColor: primaryColor,
    colorScheme: ColorScheme.light(
      primary: primaryColor,
      secondary: secondaryColor,
      background: backgroundColorLight,
    ),
    scaffoldBackgroundColor: backgroundColorLight,
    cardColor: cardColorLight,
    textTheme: TextTheme(
      bodyLarge: TextStyle(color: textColorLight),
      bodyMedium: TextStyle(color: textColorLight),
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: primaryColor,
      foregroundColor: Colors.white,
    ),
    switchTheme: SwitchThemeData(
      thumbColor: MaterialStateProperty.resolveWith<Color>((states) {
        if (states.contains(MaterialState.selected)) {
          return primaryColor;
        }
        return Colors.grey;
      }),
      trackColor: MaterialStateProperty.resolveWith<Color>((states) {
        if (states.contains(MaterialState.selected)) {
          return primaryColor.withOpacity(0.5);
        }
        return Colors.grey.withOpacity(0.5);
      }),
    ),
    sliderTheme: SliderThemeData(
      activeTrackColor: primaryColor,
      thumbColor: primaryColor,
      overlayColor: primaryColor.withOpacity(0.2),
      valueIndicatorColor: primaryColor,
    ),
  );

  /// ダークモードのテーマ
  static ThemeData darkTheme = ThemeData(
    primaryColor: primaryColor,
    colorScheme: ColorScheme.dark(
      primary: primaryColor,
      secondary: secondaryColor,
      background: backgroundColorDark,
    ),
    scaffoldBackgroundColor: backgroundColorDark,
    cardColor: cardColorDark,
    textTheme: TextTheme(
      bodyLarge: TextStyle(color: textColorDark),
      bodyMedium: TextStyle(color: textColorDark),
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: cardColorDark,
      foregroundColor: textColorDark,
    ),
    switchTheme: SwitchThemeData(
      thumbColor: MaterialStateProperty.resolveWith<Color>((states) {
        if (states.contains(MaterialState.selected)) {
          return primaryColor;
        }
        return Colors.grey;
      }),
      trackColor: MaterialStateProperty.resolveWith<Color>((states) {
        if (states.contains(MaterialState.selected)) {
          return primaryColor.withOpacity(0.5);
        }
        return Colors.grey.withOpacity(0.5);
      }),
    ),
    sliderTheme: SliderThemeData(
      activeTrackColor: primaryColor,
      thumbColor: primaryColor,
      overlayColor: primaryColor.withOpacity(0.2),
      valueIndicatorColor: primaryColor,
    ),
  );
}
