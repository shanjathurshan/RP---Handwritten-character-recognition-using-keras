import requests
import tensorflow as tf
import os
from flask import Flask, request, jsonify
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs

# 🔥 GitHub URL where your model is hosted (Replace with your actual URL)
GITHUB_MODEL_URL = "https://github.com/shanjathurshan/handwritten-character-recognition/releases/download/v1.0/my_model.h5"
LOCAL_MODEL_PATH = "/tmp/my_model.h5"  # Railway uses /tmp for temp storage

# 🔥 Download the model from GitHub if it does not exist locally
if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading model from GitHub...")
    response = requests.get(GITHUB_MODEL_URL)
    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(response.content)

# 🔥 Load the TensorFlow model
print("Loading model...")
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
print("Model loaded successfully!")

# 🔥 Initialize Flask
app = Flask(__name__)

# 🔥 Preprocessing Function (Resizes image to match model input)
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to match model input
    img = cv2.bitwise_not(img)  # Invert colors if needed
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# 🔥 Flask Route to Handle Predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = preprocess_image(file)

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[0][predicted_class]) * 100  # Convert to %

    response = {
        "predicted_character": predicted_class,
        "confidence": confidence
    }

    return jsonify(response)

# 🔥 Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
