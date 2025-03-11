import 'package:flutter/material.dart';
import 'package:hayven/constants/app_strings.dart';
import 'package:hayven/constants/app_theme.dart';
import 'package:hayven/ui/screens/home_screen.dart';
import 'package:hayven/audio/audio_provider.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AudioProvider()),
      ],
      child: MaterialApp(
        title: AppStrings.appName,
        theme: AppTheme.lightTheme,
        home: const HomeScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
