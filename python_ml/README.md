# Hayven 音声処理システム

このディレクトリには、Hayven アプリケーションの音声処理システムが含まれています。このシステムは、マイクからの音声入力を取得し、機械学習モデルを使用して赤ちゃんの泣き声を検出し、検出された場合に音量を低減します。

## 環境構築

### 必要条件

- Python 3.8 以上
- pip（Python パッケージマネージャー）

### インストール

1. セットアップスクリプトを実行して、仮想環境を作成し、必要なパッケージをインストールします。

```bash
./setup.sh
```

2. 仮想環境を有効化します。

```bash
source venv/bin/activate
```

### Apple Silicon (M1/M2/M3) Mac での注意事項

Apple Silicon を搭載した Mac では、一部のパッケージ（特に TensorFlow と tflite-support）のインストールに問題が発生する場合があります。セットアップスクリプトは自動的に対応しますが、以下の点に注意してください：

- TensorFlow は 2.13.0 以上のバージョンを使用します
- tflite-support のインストールに失敗した場合、メタデータ機能は使用できませんが、基本的なモデル変換は可能です
- モデル最適化時にエラーが発生した場合は、`optimize_model.py`のエラーメッセージを確認してください

### 依存関係の競合について

このプロジェクトでは、以下の依存関係の競合に注意が必要です：

- `asteroid`パッケージは`torch<2.0.0`を要求するため、`torch`と`torchaudio`のバージョンを 1.8.0 以上 1.9.0 未満に制限しています
- `pytorch-lightning`は`protobuf<=3.20.1`を要求するため、このバージョンを使用しています
- インストール中に依存関係の競合が発生した場合、セットアップスクリプトは各パッケージを個別にインストールして問題を回避します
- 一部のパッケージがインストールできない場合でも、基本的な機能は使用可能です

### インストールに関する追加の注意事項

- **macOS ユーザー**: `pyaudio`のインストールには`portaudio`が必要です。セットアップスクリプトは自動的に Homebrew を使用してインストールを試みますが、Homebrew がインストールされていない場合は手動でインストールする必要があります：

  ```bash
  brew install portaudio
  ```

- **pip のバージョン**: `pytorch-lightning`のメタデータ問題を回避するために、`pip<24.1`を使用しています。セットアップスクリプトは自動的に適切なバージョンをインストールします。

- **tflite-support**: 現在利用可能な最新バージョンは`0.1.0a1`（アルファ版）です。これにより一部の機能が制限される可能性があります。

## データセットの準備

### 公開データセットのダウンロード

以下のコマンドで公開データセットをダウンロードします。

```bash
python prepare_data.py --download
```

このコマンドは以下のデータセットをダウンロードします：

- ESC-50（環境音データセット）
- Crying Baby Dataset（赤ちゃんの泣き声データセット）

### データの前処理

ダウンロードしたデータセットを前処理します。

```bash
python prepare_data.py
```

このコマンドは以下の処理を行います：

1. 泣き声データセットの準備（5 秒のセグメントに分割）
2. 非泣き声データセットの準備（5 秒のセグメントに分割）
3. 泣き声と非泣き声を混合したデータセットの作成
4. メタデータファイルの作成

## モデルの学習

### カスタムモデルの学習

以下のコマンドでカスタムモデルを学習します。

```bash
python train_model.py --model_type custom
```

オプション：

- `--data_dir`: 処理済みデータのディレクトリ（デフォルト: `data/processed`）
- `--metadata`: メタデータファイル（デフォルト: `data/metadata.csv`）
- `--model_dir`: モデルの保存先ディレクトリ（デフォルト: `models/checkpoints`）
- `--epochs`: エポック数（デフォルト: 10）
- `--batch_size`: バッチサイズ（デフォルト: 16）
- `--learning_rate`: 学習率（デフォルト: 1e-4）

### Hugging Face モデルの学習

以下のコマンドで Hugging Face モデルを学習します。

```bash
python train_model.py --model_type huggingface --base_model facebook/wav2vec2-base
```

オプション：

- `--base_model`: ベースモデル（デフォルト: `facebook/wav2vec2-base`）

## モデルの最適化

学習したモデルをモバイル向けに最適化します。

```bash
python optimize_model.py --model_path models/checkpoints/crying_detector.pth --model_type custom
```

または、Hugging Face モデルの場合：

```bash
python optimize_model.py --model_path models/checkpoints/final_model --model_type huggingface
```

オプション：

- `--output_path`: 出力パス（デフォルト: `models/optimized/crying_detector_optimized.tflite`）
- `--flutter_assets`: Flutter アセットのパス（デフォルト: `../assets/models`）

## 音声処理パイプラインのテスト

最適化したモデルを使用して音声処理パイプラインをテストします。

```bash
python audio_processing/audio_pipeline.py --model models/optimized/crying_detector_optimized.tflite
```

## ディレクトリ構造

- `audio_processing/`: 音声処理関連のスクリプト
  - `audio_capture.py`: マイクからの音声キャプチャを行うクラス
  - `audio_preprocessing.py`: 音声の前処理を行うクラス
  - `audio_pipeline.py`: 音声処理パイプラインを統合するクラス
- `ml_model/`: 機械学習モデル関連のスクリプト
  - `crying_detector.py`: 赤ちゃんの泣き声を検出するクラス
- `data/`: データセットやモデルファイルを保存するディレクトリ
  - `raw/`: 生データ
  - `processed/`: 処理済みデータ
- `models/`: モデルファイルを保存するディレクトリ
  - `checkpoints/`: チェックポイント
  - `optimized/`: 最適化されたモデル
- `prepare_data.py`: データセットを準備するスクリプト
- `train_model.py`: モデルを学習するスクリプト
- `optimize_model.py`: モデルを最適化するスクリプト
- `setup.sh`: セットアップスクリプト
- `requirements.txt`: 必要なパッケージのリスト

## Flutter アプリとの連携

最適化されたモデルは自動的に Flutter アプリのアセットディレクトリ（`../assets/models`）にコピーされます。Flutter アプリからこのモデルを使用する方法については、Flutter プロジェクトのドキュメントを参照してください。
