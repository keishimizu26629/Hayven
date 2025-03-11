import numpy as np
import tensorflow as tf

class CryingDetector:
    """
    赤ちゃんの泣き声を検出するクラス
    """
    def __init__(self, model_path, threshold=0.5):
        """
        初期化

        Args:
            model_path: モデルのパス
            threshold: 検出閾値
        """
        self.threshold = threshold
        self.model = self._load_model(model_path)
        print(f"泣き声検出モデルを読み込みました: {model_path}")

    def _load_model(self, model_path):
        """
        モデルを読み込む

        Args:
            model_path: モデルのパス

        Returns:
            読み込まれたモデル
        """
        try:
            # TensorFlow Liteモデルを読み込む
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            return interpreter
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            raise

    def detect(self, audio):
        """
        泣き声を検出

        Args:
            audio: 音声データ（numpy配列）

        Returns:
            (is_crying, confidence): 泣き声かどうかのブール値と信頼度
        """
        if audio is None:
            return False, 0.0

        try:
            # 入出力テンソルの取得
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            # 入力データの準備
            input_data = np.expand_dims(audio.astype(np.float32), axis=0)

            # 入力形状の調整
            if input_data.shape[1] < input_details[0]['shape'][1]:
                # パディング
                padding = np.zeros((1, input_details[0]['shape'][1] - input_data.shape[1]))
                input_data = np.concatenate([input_data, padding], axis=1)
            elif input_data.shape[1] > input_details[0]['shape'][1]:
                # 切り取り
                input_data = input_data[:, :input_details[0]['shape'][1]]

            # 推論
            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()

            # 出力の取得
            output_data = self.model.get_tensor(output_details[0]['index'])

            # 結果の解釈
            confidence = output_data[0][1]  # クラス1（泣き声）の確率
            is_crying = confidence > self.threshold

            return is_crying, confidence

        except Exception as e:
            print(f"泣き声検出中にエラーが発生しました: {e}")
            return False, 0.0
