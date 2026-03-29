import base64
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request
from app.predict import predict_from_canvas

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save_canvas():
    data = request.json
    mode = data["mode"]
    target = data["target"]
    image_data = data["image"]

    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    canvas = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    prediction, confidence = predict_from_canvas(canvas, mode)

    is_correct = str(prediction).strip().lower() == str(target).strip().lower()

    return jsonify({
        "target": target,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "correct": is_correct
    })
if __name__ == "__main__":
    app.run(debug=True)