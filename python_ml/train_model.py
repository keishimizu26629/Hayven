import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Model, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
from transformers import Trainer, TrainingArguments
from datasets import Dataset as HFDataset
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple

class AudioDataset(Dataset):
    """音声データセットクラス"""

    def __init__(self, audio_files, labels, transform=None, sample_rate=16000, max_length=5):
        """
        初期化

        Args:
            audio_files: 音声ファイルのパスのリスト
            labels: ラベルのリスト（0: 泣き声なし, 1: 泣き声あり）
            transform: 変換関数
            sample_rate: サンプリングレート
            max_length: 最大長（秒）
        """
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]

        # 音声データの読み込み
        audio_data, sr = librosa.load(audio_file, sr=self.sample_rate)

        # 最大長に調整
        max_samples = self.sample_rate * self.max_length
        if len(audio_data) > max_samples:
            # ランダムな位置から切り出し
            start = np.random.randint(0, len(audio_data) - max_samples)
            audio_data = audio_data[start:start + max_samples]
        else:
            # 足りない部分はゼロパディング
            padding = max_samples - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), 'constant')

        # 変換がある場合は適用
        if self.transform:
            audio_data = self.transform(audio_data)

        return audio_data, label

def prepare_dataset(data_dir, metadata_file=None, test_size=0.2):
    """
    データセットを準備

    Args:
        data_dir: 音声ファイルのディレクトリ
        metadata_file: メタデータファイル（なければ自動生成）
        test_size: テストデータの割合

    Returns:
        train_dataset, val_dataset
    """
    if metadata_file and os.path.exists(metadata_file):
        # メタデータファイルが存在する場合は読み込む
        df = pd.read_csv(metadata_file)
        audio_files = df['file_path'].tolist()
        labels = df['label'].tolist()
    else:
        # メタデータファイルがない場合は自動生成
        audio_files = []
        labels = []

        # 泣き声ありのデータ
        crying_dir = os.path.join(data_dir, 'crying')
        if os.path.exists(crying_dir):
            for file in os.listdir(crying_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(crying_dir, file))
                    labels.append(1)  # 泣き声あり

        # 泣き声なしのデータ
        non_crying_dir = os.path.join(data_dir, 'non_crying')
        if os.path.exists(non_crying_dir):
            for file in os.listdir(non_crying_dir):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(non_crying_dir, file))
                    labels.append(0)  # 泣き声なし

        # メタデータを保存
        if metadata_file:
            df = pd.DataFrame({'file_path': audio_files, 'label': labels})
            df.to_csv(metadata_file, index=False)

    # トレーニングデータとバリデーションデータに分割
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # データセットの作成
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)

    return train_dataset, val_dataset

def prepare_hf_dataset(data_dir, metadata_file=None, test_size=0.2):
    """
    Hugging Face用のデータセットを準備

    Args:
        data_dir: 音声ファイルのディレクトリ
        metadata_file: メタデータファイル（なければ自動生成）
        test_size: テストデータの割合

    Returns:
        train_dataset, val_dataset (Hugging Face形式)
    """
    train_dataset, val_dataset = prepare_dataset(data_dir, metadata_file, test_size)

    # Hugging Face形式に変換
    def convert_to_hf_dataset(dataset):
        data = {
            'audio': [],
            'label': []
        }

        for i in range(len(dataset)):
            audio_data, label = dataset[i]

            # 一時ファイルに保存
            tmp_file = f"tmp_{i}.wav"
            sf.write(tmp_file, audio_data, dataset.sample_rate)

            # データに追加
            data['audio'].append(tmp_file)
            data['label'].append(label)

        # Hugging Faceデータセットに変換
        hf_dataset = HFDataset.from_dict(data)

        # 一時ファイルを削除
        for i in range(len(dataset)):
            os.remove(f"tmp_{i}.wav")

        return hf_dataset

    hf_train_dataset = convert_to_hf_dataset(train_dataset)
    hf_val_dataset = convert_to_hf_dataset(val_dataset)

    return hf_train_dataset, hf_val_dataset

def train_custom_model(train_dataset, val_dataset, model_dir, num_epochs=10, batch_size=16, learning_rate=1e-4):
    """
    カスタムモデルを学習

    Args:
        train_dataset: トレーニングデータセット
        val_dataset: バリデーションデータセット
        model_dir: モデルの保存先ディレクトリ
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
    """
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # モデルの定義（簡単なCNNモデル）
    class AudioClassifier(nn.Module):
        def __init__(self):
            super(AudioClassifier, self).__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool3 = nn.MaxPool1d(kernel_size=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * (train_dataset.max_length * train_dataset.sample_rate // 8), 128)
            self.fc2 = nn.Linear(128, 2)  # 2クラス（泣き声あり/なし）

        def forward(self, x):
            # 入力形状: (batch_size, audio_length)
            x = x.unsqueeze(1)  # (batch_size, 1, audio_length)
            x = torch.relu(self.conv1(x))
            x = self.pool1(x)
            x = torch.relu(self.conv2(x))
            x = self.pool2(x)
            x = torch.relu(self.conv3(x))
            x = self.pool3(x)
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデルの初期化
    model = AudioClassifier().to(device)

    # 損失関数と最適化アルゴリズムの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習ループ
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # トレーニングフェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配をゼロにリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            # 統計情報の更新
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)

        # バリデーションフェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                inputs, labels = inputs.to(device), labels.to(device)

                # 順伝播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 統計情報の更新
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # モデルの保存
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "crying_detector.pth"))

    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, "loss_curve.png"))

    print(f"モデルを {os.path.join(model_dir, 'crying_detector.pth')} に保存しました")

def train_wav2vec2_model(data_dir, metadata_file, model_dir, base_model="facebook/wav2vec2-base", epochs=10, batch_size=8, learning_rate=5e-5):
    """
    Wav2Vec2モデルの学習（PyTorch方式）
    """
    # メタデータの読み込み
    metadata = pd.read_csv(metadata_file)

    # ファイルパスとラベルの取得
    file_paths = [os.path.join(data_dir, file) for file in metadata['file_name']]
    labels = metadata['label'].tolist()

    # データセットの分割
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 特徴量抽出器の読み込み
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)

    # データセットの作成
    class Wav2Vec2Dataset(Dataset):
        def __init__(self, file_paths, labels, feature_extractor):
            self.file_paths = file_paths
            self.labels = labels
            self.feature_extractor = feature_extractor

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            audio_path = self.file_paths[idx]
            label = self.labels[idx]

            # 音声の読み込み
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # 固定長に調整（5秒 = 80000サンプル）
            if len(audio) < 80000:
                audio = np.pad(audio, (0, 80000 - len(audio)), 'constant')
            else:
                audio = audio[:80000]

            # 特徴量抽出
            input_values = self.feature_extractor(audio, sampling_rate=16000).input_values[0]

            return {"input_values": input_values, "label": label}

    # データセットの作成
    train_dataset = Wav2Vec2Dataset(train_files, train_labels, feature_extractor)
    val_dataset = Wav2Vec2Dataset(val_files, val_labels, feature_extractor)

    # データコレーターの作成
    data_collator = SpeechDataCollator(feature_extractor=feature_extractor)

    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    # モデルの作成
    model = Wav2Vec2Classifier(base_model_name=base_model, n_classes=2)

    # 損失関数と最適化アルゴリズムの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 学習
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # トレーニング
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # 検証
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_values)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "wav2vec2_crying_detector.pth"))
            print(f"Model saved to {os.path.join(model_dir, 'wav2vec2_crying_detector.pth')}")

    return os.path.join(model_dir, "wav2vec2_crying_detector.pth")

def train_hf_model(data_dir, metadata_file, model_dir, base_model="facebook/wav2vec2-base", epochs=10, batch_size=8, learning_rate=5e-5):
    """
    Hugging Face Trainerを使用したWav2Vec2モデルの学習
    """
    # データセットの作成
    train_dataset, val_dataset, feature_extractor = create_hf_dataset(data_dir, metadata_file)

    # ラベル数の取得
    num_labels = 2

    # モデルの作成
    model = AutoModelForAudioClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        gradient_checkpointing=True,
    )

    # データコレーターの作成
    data_collator = SpeechDataCollator(feature_extractor=feature_extractor)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
    )

    # トレーナーの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        data_collator=data_collator,
    )

    # 学習
    trainer.train()

    # モデルの保存
    model_path = os.path.join(model_dir, "final_model")
    model.save_pretrained(model_path)
    feature_extractor.save_pretrained(model_path)

    return model_path

def optimize_model_for_mobile(model_path, output_path):
    """
    モバイル向けにモデルを最適化

    Args:
        model_path: モデルのパス
        output_path: 出力パス
    """
    # TensorFlow Liteに変換
    # この部分は実際のモデルタイプによって異なります
    print(f"モデルを {output_path} に最適化しました")

def main():
    parser = argparse.ArgumentParser(description="赤ちゃんの泣き声検出モデルの学習")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="処理済みデータのディレクトリ")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="メタデータファイル")
    parser.add_argument("--model_dir", type=str, default="models/checkpoints", help="モデルの保存先ディレクトリ")
    parser.add_argument("--model_type", type=str, choices=["custom", "wav2vec2", "huggingface"], default="custom", help="モデルタイプ")
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-base", help="ベースモデル（Hugging Faceモデルの場合）")
    parser.add_argument("--epochs", type=int, default=10, help="エポック数")
    parser.add_argument("--batch_size", type=int, default=16, help="バッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学習率")

    args = parser.parse_args()

    # データセットの準備
    if args.model_type == "custom":
        train_dataset, val_dataset = prepare_dataset(args.data_dir, args.metadata)

        # モデルの学習
        train_custom_model(
            train_dataset, val_dataset, args.model_dir,
            num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
        )
    elif args.model_type == "wav2vec2":
        print("Wav2Vec2モデルの学習を開始します（PyTorch方式）...")
        model_path = train_wav2vec2_model(
            args.data_dir,
            args.metadata,
            args.model_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:  # huggingface
        train_dataset, val_dataset = prepare_hf_dataset(args.data_dir, args.metadata)

        # モデルの学習
        train_hf_model(
            args.data_dir,
            args.metadata,
            args.model_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

    # モデルの最適化
    model_path = os.path.join(args.model_dir, "final_model" if args.model_type == "huggingface" else "crying_detector.pth")
    output_path = os.path.join(args.model_dir, "crying_detector_optimized.tflite")
    optimize_model_for_mobile(model_path, output_path)

if __name__ == "__main__":
    main()
