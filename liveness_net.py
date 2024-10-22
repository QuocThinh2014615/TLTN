# liveness_net.py

from tensorflow.keras import layers
from tensorflow.keras.models import Model

class LivenessNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Định nghĩa mô hình của bạn ở đây
        input_shape = (height, width, depth)
        inputs = layers.Input(shape=input_shape)

        # Thêm các lớp vào mô hình
        # Ví dụ:
        x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Thêm các lớp khác và kết thúc mô hình
        x = layers.Flatten()(x)
        x = layers.Dense(classes)(x)
        outputs = layers.Activation("softmax")(x)

        model = Model(inputs, outputs)
        return model
