#!/bin/bash

# 仮想環境のディレクトリ
VENV_DIR="venv"

# 仮想環境が存在するか確認
if [ ! -d "$VENV_DIR" ]; then
    echo "仮想環境を作成しています..."
    python3 -m venv $VENV_DIR
else
    echo "既存の仮想環境を使用します"
fi

# 仮想環境をアクティベート
source $VENV_DIR/bin/activate

# pipをアップグレード（pytorch-lightningのメタデータ問題を回避するため古いバージョンを使用）
echo "必要なパッケージをインストールしています..."
pip install --upgrade "pip<24.1"

# macOSの場合、portaudioをインストール
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOSを検出しました。portaudioをインストールしています..."
    if command -v brew &> /dev/null; then
        brew install portaudio
    else
        echo "Homebrewがインストールされていません。pyaudioのインストールに失敗する可能性があります。"
    fi
fi

# 基本的なデータ処理ライブラリをインストール
echo "基本的なデータ処理ライブラリをインストールしています..."
pip install -r requirements_base.txt

# 音声処理ライブラリをインストール
echo "音声処理ライブラリをインストールしています..."
pip install -r requirements_audio.txt || echo "一部の音声処理ライブラリのインストールに失敗しました。"

# 深層学習ライブラリをインストール
echo "深層学習ライブラリをインストールしています..."
pip install -r requirements_ml.txt || echo "一部の深層学習ライブラリのインストールに失敗しました。"

# モデル最適化ライブラリをインストール
echo "モデル最適化ライブラリをインストールしています..."
pip install -r requirements_optimize.txt || echo "一部のモデル最適化ライブラリのインストールに失敗しました。"

# データディレクトリの作成
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p models/optimized

echo "セットアップが完了しました！"
echo "仮想環境を有効化するには以下のコマンドを実行してください："
echo "source venv/bin/activate"
