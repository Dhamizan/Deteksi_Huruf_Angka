import cv2
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import string

LOADED_MODELS = {}

MODEL_PATHS = {
    "number": {
        "path": "models/CNN_Dataset_Alphabet_Nomor.keras",
        "labels": [str(i) for i in range(10)]
    },
    "upper": {
        "path": "models/CNN_Dataset_Alphabet_Besar.keras",
        "labels": list(string.ascii_uppercase)
    },
    "lower": {
        "path": "models/CNN_Dataset_Alphabet_Kecil.keras",
        "labels": list(string.ascii_lowercase)
    }
}

def get_model(mode):
    if mode not in LOADED_MODELS:
        print(f"Loading model for {mode}...")
        LOADED_MODELS[mode] = tf.keras.models.load_model(MODEL_PATHS[mode]["path"], compile=False)
    return LOADED_MODELS[mode]

def preprocess_canvas(canvas):
    resized = cv2.resize(canvas, (64, 64))
    img_input = resized.astype("float32")
    img_input = np.expand_dims(img_input, axis=0)
    return img_input

def predict_from_canvas(canvas, mode):
    try:
        model = get_model(mode)
        labels = MODEL_PATHS[mode]["labels"]
        
        cv2.imwrite(f"images/gambar_{mode}.png", canvas)
        
        img_prepared = preprocess_canvas(canvas)
        
        preds = model.predict(img_prepared, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        
        print(f"[{mode.upper()}] Prediksi: {labels[idx]} ({round(confidence*100, 2)}%)")
        
        return labels[idx], confidence
    except Exception as e:
        print(f"Error Prediction Detail: {e}")
        return "Error", 0.0