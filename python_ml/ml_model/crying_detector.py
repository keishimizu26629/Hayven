import os
import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import pipeline
import soundfile as sf
import tempfile

class CryingDetector:
    """赤ちゃんの泣き声を検出するクラス"""

    def __init__(self, model_name="facebook/wav2vec2-base", device=None):
        """
        初期化

        Args:
            model_name: モデル名またはパス
            device: デバイス（'cpu'または'cuda'）
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"デバイス: {self.device}")

        # モデルとプロセッサを読み込む
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)

        # ラベル（カスタムモデルの場合は変更が必要）
        self.labels = ["not_crying", "crying"]

        print(f"モデル '{model_name}' を読み込みました")

    def detect_from_file(self, audio_file, threshold=0.5):
        """
        音声ファイルから泣き声を検出

        Args:
            audio_file: 音声ファイルのパス
            threshold: 検出閾値

        Returns:
            検出結果（True/False）、信頼度
        """
        # 音声ファイルを読み込む
        audio_data, sample_rate = librosa.load(audio_file, sr=None)

        return self.detect(audio_data, sample_rate, threshold)

    def detect(self, audio_data, sample_rate, threshold=0.5):
        """
        音声データから泣き声を検出

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート
            threshold: 検出閾値

        Returns:
            検出結果（True/False）、信頼度
        """
        # 一時ファイルに保存（feature_extractorが音声ファイルを必要とするため）
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)

        try:
            # 特徴量を抽出
            inputs = self.feature_extractor(
                temp_path,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)

            # 推論
            with torch.no_grad():
                outputs = self.model(**inputs)

            # ロジットを取得
            logits = outputs.logits

            # ソフトマックスで確率に変換
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # 泣き声クラスの確率を取得
            crying_prob = probs[0, self.labels.index("crying")].item()

            # 閾値と比較
            is_crying = crying_prob > threshold

            return is_crying, crying_prob

        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @classmethod
    def from_pipeline(cls, model_name="facebook/wav2vec2-base"):
        """
        Hugging Face pipelineを使用してインスタンスを作成

        Args:
            model_name: モデル名またはパス

        Returns:
            CryingDetectorインスタンス
        """
        detector = cls.__new__(cls)
        detector.device = "cuda" if torch.cuda.is_available() else "cpu"

        # pipelineを作成
        detector.pipeline = pipeline(
            "audio-classification",
            model=model_name,
            device=0 if detector.device == "cuda" else -1
        )

        # ラベル（カスタムモデルの場合は変更が必要）
        detector.labels = ["not_crying", "crying"]

        print(f"モデル '{model_name}' を読み込みました（pipeline使用）")

        return detector

    def detect_from_pipeline(self, audio_data, sample_rate, threshold=0.5):
        """
        pipelineを使用して音声データから泣き声を検出

        Args:
            audio_data: 音声データ（numpy配列）
            sample_rate: サンプリングレート
            threshold: 検出閾値

        Returns:
            検出結果（True/False）、信頼度
        """
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)

        try:
            # pipelineで推論
            result = self.pipeline(temp_path)

            # 結果を解析
            for item in result:
                if item["label"] == "crying":
                    crying_prob = item["score"]
                    is_crying = crying_prob > threshold
                    return is_crying, crying_prob

            # 泣き声クラスが見つからない場合
            return False, 0.0

        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)

# 使用例
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # テスト用の音声ファイル（存在する場合）
    test_file = "path/to/test/audio.wav"

    if os.path.exists(test_file):
        # 検出器を初期化
        detector = CryingDetector()

        # 泣き声を検出
        is_crying, confidence = detector.detect_from_file(test_file)

        print(f"泣き声検出: {'はい' if is_crying else 'いいえ'}, 信頼度: {confidence:.4f}")

        # 音声を表示
        audio_data, sample_rate = librosa.load(test_file, sr=None)
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.title("Waveform")
        plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.title("Spectrogram")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()
    else:
        print(f"テストファイル '{test_file}' が見つかりません")
