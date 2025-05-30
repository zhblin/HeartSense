
import numpy as np
import tensorflow as tf

class HeartRatePredictor:
    def __init__(self, model_path, window_size=300):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.window_size = window_size
        self.feature_buffer = []
        self.mask_buffer = []

    def update(self, new_feature, new_mask):
        self.feature_buffer.append(new_feature)
        self.mask_buffer.append(new_mask)
        if len(self.feature_buffer) > self.window_size:
            self.feature_buffer.pop(0)
            self.mask_buffer.pop(0)

    def ready(self):
        return len(self.feature_buffer) == self.window_size

    def predict(self):
        if not self.ready():
            return None
        X = np.array(self.feature_buffer).reshape(1, self.window_size, -1)
        M = np.array(self.mask_buffer).reshape(1, self.window_size, -1)
        pred = self.model.predict([X, M], verbose=0)
        return float(pred[0][0])
