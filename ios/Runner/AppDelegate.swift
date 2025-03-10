import Flutter
import UIKit
import AVFoundation

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
  private var audioEngine: AVAudioEngine?
  private var isPassthroughActive = false

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    // メソッドチャネルの設定
    let controller = window?.rootViewController as! FlutterViewController
    let methodChannel = FlutterMethodChannel(name: "com.hayven/audio_method", binaryMessenger: controller.binaryMessenger)
    let eventChannel = FlutterEventChannel(name: "com.hayven/audio_event", binaryMessenger: controller.binaryMessenger)

    // メソッドハンドラの設定
    methodChannel.setMethodCallHandler { [weak self] (call, result) in
      guard let self = self else {
        result(FlutterError(code: "UNAVAILABLE", message: "AppDelegateが利用できません", details: nil))
        return
      }

      switch call.method {
      case "startAudioPassthrough":
        self.startAudioPassthrough { success in
          result(success)
        }
      case "stopAudioPassthrough":
        self.stopAudioPassthrough()
        result(true)
      case "setVolume":
        if let args = call.arguments as? [String: Any],
           let volume = args["volume"] as? Double {
          self.setVolume(volume)
          result(true)
        } else {
          result(FlutterError(code: "INVALID_ARGUMENTS", message: "無効な引数", details: nil))
        }
      default:
        result(FlutterMethodNotImplemented)
      }
    }

    // イベントハンドラの設定
    let eventHandler = AudioEventHandler()
    eventChannel.setStreamHandler(eventHandler)

    GeneratedPluginRegistrant.register(with: self)
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  // マイク入力から音声出力への処理を開始
  private func startAudioPassthrough(completion: @escaping (Bool) -> Void) {
    if isPassthroughActive {
      completion(true)
      return
    }

    // オーディオセッションの設定
    let audioSession = AVAudioSession.sharedInstance()
    do {
      try audioSession.setCategory(.playAndRecord, options: [.allowBluetooth, .mixWithOthers])
      try audioSession.setActive(true)
    } catch {
      print("オーディオセッションの設定に失敗しました: \(error)")
      completion(false)
      return
    }

    // オーディオエンジンの設定
    audioEngine = AVAudioEngine()
    guard let audioEngine = audioEngine else {
      completion(false)
      return
    }

    // 入力と出力のノードを取得
    let inputNode = audioEngine.inputNode
    let outputNode = audioEngine.outputNode
    let mixer = AVAudioMixerNode()

    // ノードを接続
    let inputFormat = inputNode.outputFormat(forBus: 0)
    audioEngine.attach(mixer)
    audioEngine.connect(inputNode, to: mixer, format: inputFormat)
    audioEngine.connect(mixer, to: outputNode, format: inputFormat)

    // 音声処理のタップを設定
    mixer.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] (buffer, time) in
      // ここで音声処理を行う（将来的には泣き声検出などを実装）
    }

    // オーディオエンジンを開始
    do {
      try audioEngine.start()
      isPassthroughActive = true
      completion(true)
    } catch {
      print("オーディオエンジンの開始に失敗しました: \(error)")
      completion(false)
    }
  }

  // マイク入力から音声出力への処理を停止
  private func stopAudioPassthrough() {
    guard let audioEngine = audioEngine, isPassthroughActive else {
      return
    }

    // オーディオエンジンを停止
    audioEngine.stop()
    audioEngine.inputNode.removeTap(onBus: 0)

    // オーディオセッションを非アクティブにする
    do {
      try AVAudioSession.sharedInstance().setActive(false)
    } catch {
      print("オーディオセッションの非アクティブ化に失敗しました: \(error)")
    }

    isPassthroughActive = false
  }

  // 音量を設定
  private func setVolume(_ volume: Double) {
    guard let audioEngine = audioEngine, isPassthroughActive else {
      return
    }

    // 音量を設定（将来的に実装）
  }
}

// イベントハンドラ
class AudioEventHandler: NSObject, FlutterStreamHandler {
  private var eventSink: FlutterEventSink?

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    eventSink = events
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    eventSink = nil
    return nil
  }

  // イベントを送信
  func sendEvent(event: [String: Any]) {
    eventSink?(event)
  }
}
