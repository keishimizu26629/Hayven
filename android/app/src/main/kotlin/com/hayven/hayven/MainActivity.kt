package com.hayven.hayven

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val TAG = "HayvenAudio"
    private val METHOD_CHANNEL = "com.hayven/audio_method"
    private val EVENT_CHANNEL = "com.hayven/audio_event"

    private var audioProcessor: AudioProcessor? = null
    private var eventSink: EventChannel.EventSink? = null

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // メソッドチャネルの設定
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, METHOD_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "startAudioPassthrough" -> {
                    startAudioPassthrough(result)
                }
                "stopAudioPassthrough" -> {
                    stopAudioPassthrough()
                    result.success(true)
                }
                "setVolume" -> {
                    val volume = call.argument<Double>("volume")
                    if (volume != null) {
                        setVolume(volume)
                        result.success(true)
                    } else {
                        result.error("INVALID_ARGUMENTS", "無効な引数", null)
                    }
                }
                else -> {
                    result.notImplemented()
                }
            }
        }

        // イベントチャネルの設定
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, EVENT_CHANNEL).setStreamHandler(
            object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    eventSink = events
                }

                override fun onCancel(arguments: Any?) {
                    eventSink = null
                }
            }
        )
    }

    // マイク入力から音声出力への処理を開始
    private fun startAudioPassthrough(result: MethodChannel.Result) {
        if (audioProcessor != null && audioProcessor!!.isProcessing) {
            result.success(true)
            return
        }

        try {
            audioProcessor = AudioProcessor()
            audioProcessor!!.start()
            result.success(true)
        } catch (e: Exception) {
            Log.e(TAG, "オーディオ処理の開始に失敗しました: ${e.message}")
            result.error("AUDIO_ERROR", "オーディオ処理の開始に失敗しました", e.message)
        }
    }

    // マイク入力から音声出力への処理を停止
    private fun stopAudioPassthrough() {
        audioProcessor?.stop()
        audioProcessor = null
    }

    // 音量を設定
    private fun setVolume(volume: Double) {
        audioProcessor?.setVolume(volume)
    }

    // イベントを送信
    private fun sendEvent(event: Map<String, Any>) {
        activity?.runOnUiThread {
            eventSink?.success(event)
        }
    }

    // オーディオ処理クラス
    private inner class AudioProcessor {
        private val SAMPLE_RATE = 44100
        private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        var isProcessing = false
            private set
        private var audioRecord: AudioRecord? = null
        private var audioTrack: AudioTrack? = null
        private var processingThread: HandlerThread? = null
        private var handler: Handler? = null
        private var volume = 1.0

        // 処理開始
        fun start() {
            if (isProcessing) return

            // バッファサイズの計算
            val bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT
            )

            // AudioRecordの初期化
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            // AudioTrackの初期化
            audioTrack = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .setEncoding(AUDIO_FORMAT)
                        .build()
                )
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            // 処理スレッドの初期化
            processingThread = HandlerThread("AudioProcessingThread")
            processingThread!!.start()
            handler = Handler(processingThread!!.looper)

            // 処理開始
            audioRecord!!.startRecording()
            audioTrack!!.play()
            isProcessing = true

            // 処理ループ
            handler!!.post(object : Runnable {
                override fun run() {
                    if (!isProcessing) return

                    val buffer = ShortArray(bufferSize / 2)
                    val readSize = audioRecord!!.read(buffer, 0, buffer.size)

                    if (readSize > 0) {
                        // 音量調整
                        if (volume != 1.0) {
                            for (i in 0 until readSize) {
                                buffer[i] = (buffer[i] * volume).toInt().toShort()
                            }
                        }

                        // 音声出力
                        audioTrack!!.write(buffer, 0, readSize)
                    }

                    // 次のフレームを処理
                    handler!!.post(this)
                }
            })
        }

        // 処理停止
        fun stop() {
            isProcessing = false

            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            audioTrack?.stop()
            audioTrack?.release()
            audioTrack = null

            processingThread?.quitSafely()
            processingThread = null
            handler = null
        }

        // 音量設定
        fun setVolume(volume: Double) {
            this.volume = volume
        }
    }
}
