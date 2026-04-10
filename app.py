import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__, static_folder=".")
CORS(app)

model = load_model("bird_model.h5")

with open("classes.txt") as f:
    class_names = [line.strip() for line in f]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    img = keras_image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    index = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return jsonify({
        "bird": class_names[index],
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
