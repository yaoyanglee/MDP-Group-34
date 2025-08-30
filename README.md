# Custom CNN-pipeline for efficient detection-classification

---

## LOG 1 (30/08/2025)

### Results:
**Two-stage:**
'accuracy': 0.2%
'avg_latency': 0.001 ms
'mean_ram_usage': 4.4MB
'mean_cpu': 0.038 
'n': 12630
**YOLO:**
'accuracy': 99.4%
'avg_latency': 0.004 ms
'mean_ram_usage': 2.2MB
'mean_cpu': 0.925
'n': 12630

### Remarks:
- Current Detection layer:
```python
    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # use adaptive threshold for lighting robustness
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # morphology to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps_coef * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype('float32')
            warped = four_point_transform(orig, pts)
            # optionally resize to square
            h, w = warped.shape[:2]
            side = max(h, w)
            square = np.zeros((side, side, 3), dtype=warped.dtype)
            square[:h, :w] = warped
            crops.append(square)
```
- Current Classification Layer:
```python
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
```
- Changes obviously needed since Model too generic.

---
