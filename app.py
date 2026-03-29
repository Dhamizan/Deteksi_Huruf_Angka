from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from app.segmenter import segment_letters
from app.predict import predict_word
from app.validator import validate_word

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_word", methods=["POST"])
def check_word():
    data = request.json
    target = data["target"].lower()
    
    # Decode gambar base64 yang dikirim dari browser
    img_data = base64.b64decode(data["image"].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    canvas = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Logika AI kamu tetap sama
    letters = segment_letters(canvas)
    predicted = predict_word(letters)
    correct, score = validate_word(predicted, target)

    return jsonify({
        "prediction": predicted,
        "score": round(score * 100, 2),
        "correct": correct
    })

if __name__ == "__main__":
    app.run(debug=True)