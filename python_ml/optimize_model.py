import os
import argparse
import torch
import numpy as np
import tensorflow as tf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model

def convert_pytorch_to_onnx(model_path, output_path, input_shape=(1, 80000), model_type="custom"):
    """
    PyTorchモデルをONNX形式に変換

    Args:
        model_path: PyTorchモデルのパス
        output_path: 出力パス
        input_shape: 入力形状（バッチサイズ, 音声長）
        model_type: モデルタイプ（"custom"または"wav2vec2"）
    """
    # モデルの読み込み
    if model_type == "custom":
        from train_model import AudioClassifier
        model = AudioClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:  # wav2vec2
        from train_model import Wav2Vec2Classifier
        model = Wav2Vec2Classifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    # ダミー入力
    dummy_input = torch.randn(input_shape)

    # ONNX形式に変換
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"PyTorchモデルをONNX形式に変換しました: {output_path}")
    return output_path

def convert_onnx_to_tflite(onnx_path, output_path):
    """
    ONNXモデルをTensorFlow Lite形式に変換

    Args:
        onnx_path: ONNXモデルのパス
        output_path: 出力パス
    """
    # ONNXモデルをTensorFlowモデルに変換
    try:
        import onnx
        from onnx_tf.backend import prepare

        # ONNXモデルを読み込む
        onnx_model = onnx.load(onnx_path)

        # ONNXモデルをTensorFlowモデルに変換
        tf_rep = prepare(onnx_model)

        # TensorFlowモデルを保存
        tf_model_path = os.path.splitext(output_path)[0] + "_tf"
        tf_rep.export_graph(tf_model_path)

        print(f"ONNXモデルをTensorFlowモデルに変換しました: {tf_model_path}")
    except ImportError:
        # onnx_tfがインストールされていない場合はtf2onnxを使用
        import tf2onnx
        import tensorflow as tf

        # ONNXモデルをTensorFlowモデルに変換
        graph_def, inputs, outputs = tf2onnx.onnx_to_tf(onnx_path)
        tf_model = tf.function(lambda x: tf.import_graph_def(graph_def, inputs={inputs[0]: x}, return_elements=[outputs[0]])[0])

        # TensorFlowモデルを保存
        tf_model_path = os.path.splitext(output_path)[0] + "_tf"
        concrete_func = tf_model.get_concrete_function(tf.TensorSpec([1, 80000], tf.float32))
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

        # 量子化設定
        try:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

            # 代表的なデータセットを提供
            def representative_dataset():
                for _ in range(100):
                    yield [np.random.randn(1, 80000).astype(np.float32)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        except Exception as e:
            print(f"量子化設定でエラーが発生しました: {e}")
            print("基本的な変換設定を使用します")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # TensorFlow Liteモデルに変換
        tflite_model = converter.convert()

        # TensorFlow Liteモデルを保存
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"TensorFlowモデルをTensorFlow Liteモデルに変換しました: {output_path}")

        # メタデータを追加
        try:
            # tflite-supportがインストールされている場合のみメタデータを追加
            from tflite_support import metadata as _metadata
            from tflite_support import metadata_schema_py_generated as _metadata_fb

            # メタデータの作成
            model_meta = _metadata_fb.ModelMetadataT()
            model_meta.name = "CryingDetector"
            model_meta.description = "Detects baby crying sounds"
            model_meta.version = "1.0"
            model_meta.author = "Hayven"

            # 入力テンソルのメタデータ
            input_meta = _metadata_fb.TensorMetadataT()
            input_meta.name = "audio"
            input_meta.description = "Audio waveform"
            input_meta.content = _metadata_fb.ContentT()
            input_meta.content.contentProperties = _metadata_fb.AudioPropertiesT()
            input_meta.content.contentProperties.sampleRate = 16000

            # 出力テンソルのメタデータ
            output_meta = _metadata_fb.TensorMetadataT()
            output_meta.name = "probability"
            output_meta.description = "Probability of crying"

            # メタデータをモデルに追加
            model_meta.subgraphMetadata = [_metadata_fb.SubGraphMetadataT()]
            model_meta.subgraphMetadata[0].inputTensorMetadata = [input_meta]
            model_meta.subgraphMetadata[0].outputTensorMetadata = [output_meta]

            # メタデータをモデルファイルに書き込む
            b = _metadata.MetadataPopulator.with_model_file(output_path)
            b.load_metadata_buffer(_metadata.MetadataPopulator.convert_metadata_to_buffer(model_meta))
            b.populate()

            print(f"メタデータをモデルに追加しました: {output_path}")
        except ImportError as e:
            print(f"tflite-supportのインポートに失敗しました: {e}")
            print("メタデータなしでモデルを保存します")
        except Exception as e:
            print(f"メタデータの追加中にエラーが発生しました: {e}")
            print("メタデータなしでモデルを保存します")

    return output_path

def convert_hf_to_tflite(model_path, output_path):
    """
    Hugging Faceモデルを直接TensorFlow Lite形式に変換

    Args:
        model_path: Hugging Faceモデルのパス
        output_path: 出力パス
    """
    # モデルとプロセッサの読み込み
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    except:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    try:
        model = AutoModelForAudioClassification.from_pretrained(model_path)
    except:
        # カスタムWav2Vec2モデルの場合
        print("AutoModelForAudioClassificationでの読み込みに失敗しました。ONNXを経由して変換します...")
        # ONNXを経由して変換
        onnx_path = os.path.splitext(output_path)[0] + ".onnx"

        # ダミー入力
        dummy_input = torch.randn(1, 80000)

        # モデルの読み込み
        from train_model import Wav2Vec2Classifier
        model = Wav2Vec2Classifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # ONNX形式に変換
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"Wav2Vec2モデルをONNX形式に変換しました: {onnx_path}")

        # ONNXモデルをTensorFlow Liteモデルに変換
        return convert_onnx_to_tflite(onnx_path, output_path)

    # TensorFlowモデルに変換
    try:
        # TensorFlow Liteコンバーターの設定
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

        # 最適化設定
        try:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        except Exception as e:
            print(f"最適化設定でエラーが発生しました: {e}")
            print("基本的な変換設定を使用します")

        # TensorFlow Liteモデルに変換
        tflite_model = converter.convert()

        # TensorFlow Liteモデルを保存
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Hugging Faceモデルを直接TensorFlow Liteモデルに変換しました: {output_path}")

        # メタデータを追加
        try:
            # tflite-supportがインストールされている場合のみメタデータを追加
            from tflite_support import metadata as _metadata
            from tflite_support import metadata_schema_py_generated as _metadata_fb

            # メタデータの作成
            model_meta = _metadata_fb.ModelMetadataT()
            model_meta.name = "CryingDetector"
            model_meta.description = "Detects baby crying sounds"
            model_meta.version = "1.0"
            model_meta.author = "Hayven"

            # 入力テンソルのメタデータ
            input_meta = _metadata_fb.TensorMetadataT()
            input_meta.name = "audio"
            input_meta.description = "Audio waveform"
            input_meta.content = _metadata_fb.ContentT()
            input_meta.content.contentProperties = _metadata_fb.AudioPropertiesT()
            input_meta.content.contentProperties.sampleRate = 16000

            # 出力テンソルのメタデータ
            output_meta = _metadata_fb.TensorMetadataT()
            output_meta.name = "probability"
            output_meta.description = "Probability of crying"

            # メタデータをモデルに追加
            model_meta.subgraphMetadata = [_metadata_fb.SubGraphMetadataT()]
            model_meta.subgraphMetadata[0].inputTensorMetadata = [input_meta]
            model_meta.subgraphMetadata[0].outputTensorMetadata = [output_meta]

            # メタデータをモデルファイルに書き込む
            b = _metadata.MetadataPopulator.with_model_file(output_path)
            b.load_metadata_buffer(_metadata.MetadataPopulator.convert_metadata_to_buffer(model_meta))
            b.populate()

            print(f"メタデータをモデルに追加しました: {output_path}")
        except ImportError as e:
            print(f"tflite-supportのインポートに失敗しました: {e}")
            print("メタデータなしでモデルを保存します")
        except Exception as e:
            print(f"メタデータの追加中にエラーが発生しました: {e}")
            print("メタデータなしでモデルを保存します")
    except Exception as e:
        print(f"Hugging Faceモデルの変換中にエラーが発生しました: {e}")
        print("ONNXを経由して変換を試みます...")

        # ONNXを経由して変換
        onnx_path = os.path.splitext(output_path)[0] + ".onnx"

        # ダミー入力
        dummy_input = torch.randn(1, 80000)

        # ONNX形式に変換
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"Hugging FaceモデルをONNX形式に変換しました: {onnx_path}")

        # ONNXモデルをTensorFlow Liteモデルに変換
        convert_onnx_to_tflite(onnx_path, output_path)

    return output_path

def copy_to_flutter_assets(tflite_path, flutter_assets_path):
    """
    TensorFlow LiteモデルをFlutterアセットディレクトリにコピー

    Args:
        tflite_path: TensorFlow Liteモデルのパス
        flutter_assets_path: Flutterアセットディレクトリのパス
    """
    import shutil

    # ディレクトリが存在しない場合は作成
    os.makedirs(flutter_assets_path, exist_ok=True)

    # モデルをコピー
    dest_path = os.path.join(flutter_assets_path, os.path.basename(tflite_path))
    shutil.copy2(tflite_path, dest_path)

    print(f"モデルをFlutterアセットディレクトリにコピーしました: {dest_path}")
    return dest_path

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='モデルを最適化')
    parser.add_argument('--model_path', type=str, required=True, help='モデルのパス')
    parser.add_argument('--model_type', type=str, choices=['custom', 'wav2vec2', 'huggingface'], default='custom', help='モデルの種類')
    parser.add_argument('--output_path', type=str, default='models/optimized/crying_detector_optimized.tflite', help='出力パス')
    parser.add_argument('--flutter_assets', type=str, default='../assets/models', help='Flutterアセットのパス')
    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # モデルの最適化
    if args.model_type == 'custom':
        # PyTorchモデルをONNX形式に変換
        onnx_path = os.path.splitext(args.output_path)[0] + '.onnx'
        convert_pytorch_to_onnx(args.model_path, onnx_path, model_type="custom")

        # ONNXモデルをTensorFlow Lite形式に変換
        convert_onnx_to_tflite(onnx_path, args.output_path)
    elif args.model_type == 'wav2vec2':
        # PyTorchモデルをONNX形式に変換
        onnx_path = os.path.splitext(args.output_path)[0] + '.onnx'
        convert_pytorch_to_onnx(args.model_path, onnx_path, model_type="wav2vec2")

        # ONNXモデルをTensorFlow Lite形式に変換
        convert_onnx_to_tflite(onnx_path, args.output_path)
    else:
        # Hugging FaceモデルをTensorFlow Lite形式に変換
        convert_hf_to_tflite(args.model_path, args.output_path)

    # TensorFlow LiteモデルをFlutterアセットディレクトリにコピー
    copy_to_flutter_assets(args.output_path, args.flutter_assets)

if __name__ == '__main__':
    main()
