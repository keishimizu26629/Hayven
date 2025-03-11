import os
import argparse
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random
import shutil
from pydub import AudioSegment

def download_datasets(output_dir):
    """
    公開データセットをダウンロード

    Args:
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # ESC-50データセット（環境音）
    # https://github.com/karolpiczak/ESC-50
    print("ESC-50データセットをダウンロードしています...")
    os.system(f"wget -q -O {output_dir}/ESC-50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip")
    os.system(f"unzip -q -o {output_dir}/ESC-50.zip -d {output_dir}")
    os.system(f"mv {output_dir}/ESC-50-master/audio {output_dir}/ESC-50")
    os.system(f"rm -rf {output_dir}/ESC-50-master {output_dir}/ESC-50.zip")

    # Crying Baby Datasetをダウンロード
    # https://github.com/gveres/donateacry-corpus
    print("Crying Baby Datasetをダウンロードしています...")
    os.system(f"wget -q -O {output_dir}/donateacry.zip https://github.com/gveres/donateacry-corpus/archive/refs/heads/master.zip")
    os.system(f"unzip -q -o {output_dir}/donateacry.zip -d {output_dir}")
    os.system(f"mv {output_dir}/donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data {output_dir}/crying_baby")
    os.system(f"rm -rf {output_dir}/donateacry-corpus-master {output_dir}/donateacry.zip")

    print("データセットのダウンロードが完了しました")

def prepare_crying_dataset(crying_dir, output_dir, sample_rate=16000, duration=5):
    """
    泣き声データセットを準備

    Args:
        crying_dir: 泣き声データセットのディレクトリ
        output_dir: 出力ディレクトリ
        sample_rate: サンプリングレート
        duration: 音声の長さ（秒）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 出力ディレクトリ
    crying_output_dir = os.path.join(output_dir, "crying")
    os.makedirs(crying_output_dir, exist_ok=True)

    # 音声ファイルのリスト
    audio_files = []
    for root, _, files in os.walk(crying_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".ogg")):
                audio_files.append(os.path.join(root, file))

    print(f"泣き声データセット: {len(audio_files)}ファイル")

    # 音声ファイルを処理
    for i, file in enumerate(tqdm(audio_files, desc="泣き声データセットを処理中")):
        try:
            # 音声データの読み込み
            audio_data, sr = librosa.load(file, sr=sample_rate)

            # 音声の長さを調整
            if len(audio_data) > sample_rate * duration:
                # 複数のセグメントに分割
                num_segments = len(audio_data) // (sample_rate * duration)
                for j in range(num_segments):
                    start = j * sample_rate * duration
                    end = start + sample_rate * duration
                    segment = audio_data[start:end]

                    # 保存
                    output_file = os.path.join(crying_output_dir, f"crying_{i}_{j}.wav")
                    sf.write(output_file, segment, sample_rate)
            else:
                # 音声が短い場合はそのまま保存
                output_file = os.path.join(crying_output_dir, f"crying_{i}.wav")
                sf.write(output_file, audio_data, sample_rate)
        except Exception as e:
            print(f"ファイル {file} の処理中にエラーが発生しました: {e}")

    print(f"泣き声データセットの処理が完了しました")

def prepare_non_crying_dataset(non_crying_dir, output_dir, sample_rate=16000, duration=5):
    """
    非泣き声データセットを準備

    Args:
        non_crying_dir: 非泣き声データセットのディレクトリ
        output_dir: 出力ディレクトリ
        sample_rate: サンプリングレート
        duration: 音声の長さ（秒）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 出力ディレクトリ
    non_crying_output_dir = os.path.join(output_dir, "non_crying")
    os.makedirs(non_crying_output_dir, exist_ok=True)

    # 音声ファイルのリスト
    audio_files = []
    for root, _, files in os.walk(non_crying_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".ogg")):
                audio_files.append(os.path.join(root, file))

    print(f"非泣き声データセット: {len(audio_files)}ファイル")

    # 音声ファイルを処理
    for i, file in enumerate(tqdm(audio_files, desc="非泣き声データセットを処理中")):
        try:
            # 音声データの読み込み
            audio_data, sr = librosa.load(file, sr=sample_rate)

            # 音声の長さを調整
            if len(audio_data) > sample_rate * duration:
                # 複数のセグメントに分割
                num_segments = len(audio_data) // (sample_rate * duration)
                for j in range(num_segments):
                    start = j * sample_rate * duration
                    end = start + sample_rate * duration
                    segment = audio_data[start:end]

                    # 保存
                    output_file = os.path.join(non_crying_output_dir, f"non_crying_{i}_{j}.wav")
                    sf.write(output_file, segment, sample_rate)
            else:
                # 音声が短い場合はそのまま保存
                output_file = os.path.join(non_crying_output_dir, f"non_crying_{i}.wav")
                sf.write(output_file, audio_data, sample_rate)
        except Exception as e:
            print(f"ファイル {file} の処理中にエラーが発生しました: {e}")

    print(f"非泣き声データセットの処理が完了しました")

def create_mixed_dataset(crying_dir, non_crying_dir, output_dir, num_samples=100, sample_rate=16000, duration=5):
    """
    泣き声と非泣き声を混合したデータセットを作成

    Args:
        crying_dir: 泣き声データセットのディレクトリ
        non_crying_dir: 非泣き声データセットのディレクトリ
        output_dir: 出力ディレクトリ
        num_samples: サンプル数
        sample_rate: サンプリングレート
        duration: 音声の長さ（秒）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 出力ディレクトリ
    mixed_output_dir = os.path.join(output_dir, "mixed")
    os.makedirs(mixed_output_dir, exist_ok=True)

    # 泣き声ファイルのリスト
    crying_files = []
    for root, _, files in os.walk(crying_dir):
        for file in files:
            if file.endswith(".wav"):
                crying_files.append(os.path.join(root, file))

    # 非泣き声ファイルのリスト
    non_crying_files = []
    for root, _, files in os.walk(non_crying_dir):
        for file in files:
            if file.endswith(".wav"):
                non_crying_files.append(os.path.join(root, file))

    print(f"泣き声ファイル: {len(crying_files)}ファイル")
    print(f"非泣き声ファイル: {len(non_crying_files)}ファイル")

    # 混合データセットを作成
    for i in tqdm(range(num_samples), desc="混合データセットを作成中"):
        try:
            # ランダムに泣き声ファイルと非泣き声ファイルを選択
            crying_file = random.choice(crying_files)
            non_crying_file = random.choice(non_crying_files)

            # 音声データの読み込み
            crying_audio = AudioSegment.from_wav(crying_file)
            non_crying_audio = AudioSegment.from_wav(non_crying_file)

            # 音量を調整
            crying_volume = random.uniform(-10, 0)  # -10dB〜0dB
            non_crying_volume = random.uniform(-5, 5)  # -5dB〜5dB
            crying_audio = crying_audio + crying_volume
            non_crying_audio = non_crying_audio + non_crying_volume

            # 音声の長さを調整
            target_length_ms = duration * 1000
            if len(crying_audio) > target_length_ms:
                start = random.randint(0, len(crying_audio) - target_length_ms)
                crying_audio = crying_audio[start:start + target_length_ms]
            else:
                # 足りない部分は無音で埋める
                silence = AudioSegment.silent(duration=target_length_ms - len(crying_audio))
                crying_audio = crying_audio + silence

            if len(non_crying_audio) > target_length_ms:
                start = random.randint(0, len(non_crying_audio) - target_length_ms)
                non_crying_audio = non_crying_audio[start:start + target_length_ms]
            else:
                # 足りない部分は無音で埋める
                silence = AudioSegment.silent(duration=target_length_ms - len(non_crying_audio))
                non_crying_audio = non_crying_audio + silence

            # 混合
            mixed_audio = non_crying_audio.overlay(crying_audio)

            # 保存
            output_file = os.path.join(mixed_output_dir, f"mixed_{i}.wav")
            mixed_audio.export(output_file, format="wav")

            # メタデータに追加
            with open(os.path.join(output_dir, "mixed_metadata.csv"), "a") as f:
                f.write(f"{output_file},1,{crying_file},{non_crying_file},{crying_volume},{non_crying_volume}\n")

        except Exception as e:
            print(f"混合データセットの作成中にエラーが発生しました: {e}")

    print(f"混合データセットの作成が完了しました")

def create_metadata(data_dir, output_file):
    """
    メタデータファイルを作成

    Args:
        data_dir: データディレクトリ
        output_file: 出力ファイル
    """
    # 泣き声ディレクトリ
    crying_dir = os.path.join(data_dir, "crying")

    # 非泣き声ディレクトリ
    non_crying_dir = os.path.join(data_dir, "non_crying")

    # 混合ディレクトリ
    mixed_dir = os.path.join(data_dir, "mixed")

    # メタデータを作成
    metadata = []

    # 泣き声ファイル
    if os.path.exists(crying_dir):
        for file in os.listdir(crying_dir):
            if file.endswith(".wav"):
                metadata.append({
                    "file_path": os.path.join(crying_dir, file),
                    "label": 1,  # 泣き声あり
                })

    # 非泣き声ファイル
    if os.path.exists(non_crying_dir):
        for file in os.listdir(non_crying_dir):
            if file.endswith(".wav"):
                metadata.append({
                    "file_path": os.path.join(non_crying_dir, file),
                    "label": 0,  # 泣き声なし
                })

    # 混合ファイル
    if os.path.exists(mixed_dir):
        for file in os.listdir(mixed_dir):
            if file.endswith(".wav"):
                metadata.append({
                    "file_path": os.path.join(mixed_dir, file),
                    "label": 1,  # 泣き声あり
                })

    # DataFrameに変換
    df = pd.DataFrame(metadata)

    # 保存
    df.to_csv(output_file, index=False)

    print(f"メタデータを {output_file} に保存しました（{len(df)}件）")

def main():
    parser = argparse.ArgumentParser(description="データセットの準備")
    parser.add_argument("--download", action="store_true", help="データセットをダウンロード")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="生データのディレクトリ")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="処理済みデータのディレクトリ")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="メタデータファイル")
    parser.add_argument("--sample_rate", type=int, default=16000, help="サンプリングレート")
    parser.add_argument("--duration", type=int, default=5, help="音声の長さ（秒）")
    parser.add_argument("--mixed_samples", type=int, default=100, help="混合サンプル数")

    args = parser.parse_args()

    # ディレクトリの作成
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata), exist_ok=True)

    # データセットのダウンロード
    if args.download:
        download_datasets(args.raw_dir)

    # 泣き声データセットの準備
    crying_dir = os.path.join(args.raw_dir, "crying_baby")
    if os.path.exists(crying_dir):
        prepare_crying_dataset(crying_dir, args.processed_dir, args.sample_rate, args.duration)
    else:
        print(f"泣き声データセットが見つかりません: {crying_dir}")

    # 非泣き声データセットの準備
    non_crying_dir = os.path.join(args.raw_dir, "ESC-50")
    if os.path.exists(non_crying_dir):
        prepare_non_crying_dataset(non_crying_dir, args.processed_dir, args.sample_rate, args.duration)
    else:
        print(f"非泣き声データセットが見つかりません: {non_crying_dir}")

    # 混合データセットの作成
    crying_processed_dir = os.path.join(args.processed_dir, "crying")
    non_crying_processed_dir = os.path.join(args.processed_dir, "non_crying")
    if os.path.exists(crying_processed_dir) and os.path.exists(non_crying_processed_dir):
        create_mixed_dataset(
            crying_processed_dir, non_crying_processed_dir, args.processed_dir,
            args.mixed_samples, args.sample_rate, args.duration
        )

    # メタデータの作成
    create_metadata(args.processed_dir, args.metadata)

if __name__ == "__main__":
    main()
