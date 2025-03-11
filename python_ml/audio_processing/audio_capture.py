import numpy as np
import sounddevice as sd
import threading
import queue
import time

class AudioCapture:
    """
    マイクからの音声キャプチャを行うクラス
    """
    def __init__(self, sample_rate=16000, chunk_size=8000, channels=1):
        """
        初期化

        Args:
            sample_rate: サンプリングレート（Hz）
            chunk_size: チャンクサイズ（サンプル数）
            channels: チャンネル数
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.thread = None

    def start(self):
        """
        音声キャプチャを開始
        """
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_audio)
        self.thread.daemon = True
        self.thread.start()
        print("音声キャプチャを開始しました")

    def stop(self):
        """
        音声キャプチャを停止
        """
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("音声キャプチャを停止しました")

    def _capture_audio(self):
        """
        音声キャプチャのメインループ
        """
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            # モノラルに変換
            if self.channels == 1:
                data = indata[:, 0]
            else:
                data = indata
            self.audio_queue.put(data.copy())

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                              callback=callback, blocksize=self.chunk_size):
                while self.is_running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"音声キャプチャ中にエラーが発生しました: {e}")
            self.is_running = False

    def get_audio(self, timeout=1):
        """
        キューから音声データを取得

        Args:
            timeout: タイムアウト（秒）

        Returns:
            音声データ（numpy配列）
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# 使用例
if __name__ == "__main__":
    audio_capture = AudioCapture()

    try:
        audio_capture.start()
        print("5秒間音声をキャプチャします...")
        time.sleep(5)
    finally:
        audio_capture.stop()
