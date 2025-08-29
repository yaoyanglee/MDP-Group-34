"""Tiny CNN classifier using TensorFlow/Keras.

API:
 - TinyClassifier(num_classes, input_size)
   - .build() -> compiles model
   - .fit(train_ds, val_ds, epochs)
   - .predict_batch(images) -> np.array of logits/probs
   - .save(path), .load(path)
"""

from typing import Tuple, Sequence
import numpy as np
import tensorflow as tf
keras = tf.keras



class TinyClassifier:
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (64, 64), lr: float = 1e-3):
        self.num_classes = int(num_classes)
        # Ensure input_size is always Tuple[int, int]
        if isinstance(input_size, tuple) and len(input_size) == 2:
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            raise ValueError("input_size must be a tuple of two ints")
        self.lr = lr
        self.model = None

    def build(self):
        inputs = keras.Input(shape=(*self.input_size, 3))
        x = keras.layers.Rescaling(1.0 / 255)(inputs)
        x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs, name='tiny_classifier')
        model.compile(optimizer=keras.optimizers.Adam(self.lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model

    def fit(self, x, y, val=None, epochs: int = 10):
        if self.model is None:
            self.build()
        if self.model is None:
            raise RuntimeError('Model not built or loaded')
        val_data = None
        if val is not None:
            val_data = (val[0], val[1])
        return self.model.fit(x, y, validation_data=val_data, epochs=epochs)

    def predict_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('Model not built or loaded')
        # Ensure input_size is Tuple[int, int]
        arr = np.array([cv2_resize(img, (self.input_size[0], self.input_size[1])) for img in images])
        if arr.ndim != 4 or arr.shape[1:3] != self.input_size:
            print(f"[ERROR] Input batch shape {arr.shape} does not match expected {(None, *self.input_size, 3)}")
            raise ValueError(f"Input batch shape {arr.shape} does not match expected {(None, *self.input_size, 3)}")
        print(f"[DEBUG] Classifier batch predict: batch size {arr.shape[0]}, input size {arr.shape[1:3]}")
        return self.model.predict(arr)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError('Model not built or loaded')
        self.model.save(path)

    def load(self, path: str):
        self.model = keras.models.load_model(path)
        return self.model


def cv2_resize(img: np.ndarray, size: Tuple[int, int]):
    import cv2
    # Ensure size is Tuple[int, int]
    if not (isinstance(size, tuple) and len(size) == 2):
        raise ValueError("size must be a tuple of two ints")
    h, w = int(size[0]), int(size[1])
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = np.stack([resized] * 3, axis=-1)
    return resized
