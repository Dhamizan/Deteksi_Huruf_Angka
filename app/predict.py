import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
import string

MODEL_PATH = "models/CNN_Dataset_Alphabet_Kecil.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = list(string.ascii_lowercase)

# Ini digunakan untuk pra-pemrosesan gambar sebelum prediksi
def preprocess(img):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    return np.expand_dims(img, axis=0)

# Ini digunakan untuk memprediksi kata dari daftar gambar huruf
def predict_word(letters):
    result = ""

    for i, letter_img in enumerate(letters):
        cv2.imwrite(f"images/gambar_{i}.png", letter_img)
        x = preprocess(letter_img)
        pred = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(pred))
        print(f"Letter {i}: pred={pred}, idx={idx}, label={labels[idx]}")
        result += labels[idx]

    return result

