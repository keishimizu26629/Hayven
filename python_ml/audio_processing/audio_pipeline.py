import os
import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import queue
import time

# 自作モジュールのインポート
from audio_processing.audio_capture import AudioCapture
from audio_processing.audio_preprocessing import AudioPreprocessor
from audio_processing.crying_detector import CryingDetector
from audio_processing.audio_separator import AudioSeparator

class AudioPipeline:
    """
    音声処理パイプラインを統合するクラス
    """
    def __init__(self, model_path, threshold=0.5, sample_rate=16000, chunk_size=8000):
        """
        初期化

        Args:
            model_path: モデルのパス
            threshold: 検出閾値
            sample_rate: サンプリングレート（Hz）
            chunk_size: チャンクサイズ（サンプル数）
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # コンポーネントの初期化
        self.audio_capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size)
        self.audio_preprocessing = AudioPreprocessor(sample_rate=sample_rate, frame_length=chunk_size)
        self.crying_detector = CryingDetector(model_path=model_path, threshold=threshold)
        self.audio_separator = AudioSeparator(sample_rate=sample_rate)

        # 状態
        self.is_running = False
        self.is_crying_detected = False
        self.detection_confidence = 0.0

        # 入出力キュー
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # 出力ストリーム
        self.output_stream = None

    def start(self):
        """
        パイプラインを開始
        """
        self.is_running = True

        # 音声キャプチャを開始
        self.audio_capture.start()

        # 出力ストリームを開始
        self.output_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size // 2  # 50%オーバーラップに合わせる
        )
        self.output_stream.start()

        print("音声処理パイプラインを開始しました。Ctrl+Cで終了します。")

        try:
            while self.is_running:
                # 音声データを取得
                audio = self.audio_capture.get_audio()

                if audio is None:
                    continue

                # 前処理
                processed_audio = self.audio_preprocessing.process(audio)

                if processed_audio is not None:
                    # 泣き声検出
                    is_crying, confidence = self.crying_detector.detect(processed_audio)

                    # 状態を更新
                    self.is_crying_detected = is_crying
                    self.detection_confidence = confidence

                    # 結果を表示
                    if is_crying:
                        print(f"泣き声を検出しました（信頼度: {confidence:.2f}）")

                    # 音声分離
                    separated_audio = self.audio_separator.process(processed_audio, is_crying, confidence)

                    # 出力
                    self.output_stream.write(separated_audio)

        except KeyboardInterrupt:
            print("パイプラインを終了します...")
        finally:
            self.stop()

    def stop(self):
        """
        パイプラインを停止
        """
        self.is_running = False

        # 音声キャプチャを停止
        self.audio_capture.stop()

        # 出力ストリームを停止
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()

        # 状態をリセット
        self.audio_preprocessing.reset()
        self.audio_separator.reset()

        print("パイプラインを停止しました")

    def get_state(self):
        """
        現在の状態を取得

        Returns:
            状態を表す辞書
        """
        return {
            "is_running": self.is_running,
            "is_crying_detected": self.is_crying_detected,
            "detection_confidence": self.detection_confidence
        }

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="赤ちゃんの泣き声検出・除去パイプライン")
    parser.add_argument("--model", type=str, default="models/optimized/crying_detector_optimized.tflite", help="モデルのパス")
    parser.add_argument("--threshold", type=float, default=0.5, help="検出閾値")
    parser.add_argument("--sample_rate", type=int, default=16000, help="サンプルレート")
    parser.add_argument("--chunk_size", type=int, default=8000, help="チャンクサイズ")
    args = parser.parse_args()

    # パイプラインの初期化と開始
    pipeline = AudioPipeline(
        model_path=args.model,
        threshold=args.threshold,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size
    )
    pipeline.start()

if __name__ == "__main__":
    main()
