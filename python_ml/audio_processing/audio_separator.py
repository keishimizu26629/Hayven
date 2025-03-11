import numpy as np
from scipy import signal

class AudioSeparator:
    """
    音声分離を行うクラス（赤ちゃんの泣き声を除去）
    """
    def __init__(self, sample_rate=16000, fade_duration=0.01):
        """
        初期化

        Args:
            sample_rate: サンプリングレート（Hz）
            fade_duration: フェード時間（秒）
        """
        self.sample_rate = sample_rate
        self.fade_samples = int(fade_duration * sample_rate)
        self.prev_frame = None
        self.prev_is_crying = False
        self.prev_confidence = 0.0

    def process(self, audio, is_crying, confidence):
        """
        音声分離処理

        Args:
            audio: 音声データ（numpy配列）
            is_crying: 泣き声かどうか
            confidence: 信頼度

        Returns:
            処理済みの音声データ（numpy配列）
        """
        if audio is None:
            return None

        # 泣き声の場合は音量を下げる（完全に消すのではなく）
        if is_crying:
            # 信頼度に応じて音量を調整（信頼度が高いほど音量を下げる）
            volume_reduction = min(0.9, confidence * 0.9)  # 最大90%の音量低減
            processed_audio = audio * (1.0 - volume_reduction)

            # 前のフレームが泣き声でなかった場合はフェードインを適用
            if not self.prev_is_crying and self.prev_frame is not None:
                fade_in = np.linspace(1.0, 1.0 - volume_reduction, self.fade_samples)
                processed_audio[:self.fade_samples] = audio[:self.fade_samples] * fade_in
        else:
            processed_audio = audio

            # 前のフレームが泣き声だった場合はフェードアウトを適用
            if self.prev_is_crying and self.prev_frame is not None:
                # 前のフレームの音量低減率を計算
                prev_volume_reduction = min(0.9, self.prev_confidence * 0.9)
                fade_out = np.linspace(1.0 - prev_volume_reduction, 1.0, self.fade_samples)
                processed_audio[:self.fade_samples] = audio[:self.fade_samples] * fade_out

        # 状態を更新
        self.prev_frame = audio
        self.prev_is_crying = is_crying
        self.prev_confidence = confidence

        return processed_audio

    def apply_bandpass_filter(self, audio, lowcut=300, highcut=3000, order=5):
        """
        バンドパスフィルタを適用

        Args:
            audio: 音声データ（numpy配列）
            lowcut: 低域カットオフ周波数（Hz）
            highcut: 高域カットオフ周波数（Hz）
            order: フィルタの次数

        Returns:
            フィルタ適用後の音声データ（numpy配列）
        """
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)

    def apply_notch_filter(self, audio, notch_freq, quality_factor=30):
        """
        ノッチフィルタを適用（特定の周波数を除去）

        Args:
            audio: 音声データ（numpy配列）
            notch_freq: 除去する周波数（Hz）
            quality_factor: クオリティファクタ

        Returns:
            フィルタ適用後の音声データ（numpy配列）
        """
        nyq = 0.5 * self.sample_rate
        freq = notch_freq / nyq
        b, a = signal.iirnotch(freq, quality_factor)
        return signal.filtfilt(b, a, audio)

    def reset(self):
        """
        状態をリセット
        """
        self.prev_frame = None
        self.prev_is_crying = False
        self.prev_confidence = 0.0
