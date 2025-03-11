import numpy as np
import librosa
import torch

class AudioPreprocessor:
    """音声の前処理を行うクラス"""

    def __init__(self, sample_rate=16000, frame_length=8000, hop_length=4000):
        """
        初期化

        Args:
            sample_rate: サンプリングレート（Hz）
            frame_length: フレーム長（サンプル数）
            hop_length: ホップ長（サンプル数）
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.buffer = np.zeros(0)

    def process(self, audio):
        """
        音声の前処理

        Args:
            audio: 音声データ（numpy配列）

        Returns:
            前処理済みの音声データ（numpy配列）
        """
        if audio is None:
            return None

        # バッファに追加
        self.buffer = np.concatenate([self.buffer, audio])

        # フレーム長に達していない場合はNoneを返す
        if len(self.buffer) < self.frame_length:
            return None

        # フレームを取得
        frame = self.buffer[:self.frame_length]

        # バッファを更新（ホップ長分だけ進める）
        self.buffer = self.buffer[self.hop_length:]

        # 前処理
        # 1. 正規化
        frame = self._normalize(frame)

        # 2. プリエンファシス
        frame = self._pre_emphasis(frame)

        return frame

    def _normalize(self, audio):
        """
        音声を正規化

        Args:
            audio: 音声データ（numpy配列）

        Returns:
            正規化された音声データ（numpy配列）
        """
        # 最大振幅で正規化
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            return audio / max_abs
        return audio

    def _pre_emphasis(self, audio, coef=0.97):
        """
        プリエンファシスフィルタを適用

        Args:
            audio: 音声データ（numpy配列）
            coef: フィルタ係数

        Returns:
            フィルタ適用後の音声データ（numpy配列）
        """
        # 高周波成分を強調
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    def reset(self):
        """
        バッファをリセット
        """
        self.buffer = np.zeros(0)

    def extract_features(self, audio_data, feature_type='melspectrogram'):
        """
        音声データから特徴量を抽出

        Args:
            audio_data: 音声データ（numpy配列）
            feature_type: 特徴量の種類（'melspectrogram', 'mfcc', 'stft'）

        Returns:
            特徴量（numpy配列）
        """
        if feature_type == 'melspectrogram':
            # メルスペクトログラムを計算
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data.flatten(),
                sr=self.sample_rate,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                n_mels=128
            )
            # 対数変換
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec

        elif feature_type == 'mfcc':
            # MFCCを計算
            mfcc = librosa.feature.mfcc(
                y=audio_data.flatten(),
                sr=self.sample_rate,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                n_mfcc=128
            )
            return mfcc

        elif feature_type == 'stft':
            # STFTを計算
            stft = librosa.stft(
                y=audio_data.flatten(),
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            # 振幅スペクトログラムを計算
            magnitude = np.abs(stft)
            # 対数変換
            log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
            return log_magnitude

        else:
            raise ValueError(f"未対応の特徴量タイプ: {feature_type}")

    def normalize_features(self, features, mean=None, std=None):
        """
        特徴量を正規化

        Args:
            features: 特徴量（numpy配列）
            mean: 平均値（Noneの場合は特徴量から計算）
            std: 標準偏差（Noneの場合は特徴量から計算）

        Returns:
            正規化された特徴量（numpy配列）、平均値、標準偏差
        """
        if mean is None:
            mean = np.mean(features)

        if std is None:
            std = np.std(features)
            # ゼロ除算を防ぐ
            if std == 0:
                std = 1e-10

        normalized_features = (features - mean) / std
        return normalized_features, mean, std

    def prepare_for_model(self, features, device='cpu'):
        """
        特徴量をモデル入力用に準備

        Args:
            features: 特徴量（numpy配列）
            device: デバイス（'cpu'または'cuda'）

        Returns:
            モデル入力用のテンソル
        """
        # バッチ次元とチャンネル次元を追加
        features = np.expand_dims(np.expand_dims(features, axis=0), axis=0)
        # NumPy配列からPyTorchテンソルに変換
        tensor = torch.from_numpy(features).float().to(device)
        return tensor

# 使用例
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # テスト用の音声データを生成
    duration = 2  # 秒
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440Hzの正弦波 + ノイズ
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

    # 前処理器を初期化
    preprocessor = AudioPreprocessor(sample_rate=sample_rate)

    # メルスペクトログラムを抽出
    mel_spec = preprocessor.extract_features(audio_data, feature_type='melspectrogram')

    # 正規化
    normalized_mel_spec, mean, std = preprocessor.normalize_features(mel_spec)

    # 結果を表示
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Mel Spectrogram")
    librosa.display.specshow(
        mel_spec,
        sr=sample_rate,
        hop_length=preprocessor.hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 2, 2)
    plt.title("Normalized Mel Spectrogram")
    librosa.display.specshow(
        normalized_mel_spec,
        sr=sample_rate,
        hop_length=preprocessor.hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar()

    plt.tight_layout()
    plt.show()
