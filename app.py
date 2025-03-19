from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import os
from werkzeug.utils import secure_filename

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress unnecessary logs

app = Flask(__name__)

# Load trained model
MODEL_PATH = "my_model.h5"
model = load_model(MODEL_PATH)

# Define labels
label_dictionary = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
    45: 'r', 46: 't'
}

# Image Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_not(img)  # Invert colors if needed
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    # Preprocess the image
    img = preprocess_image(filepath)

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[0][predicted_class]) * 100  # Convert to %

    # Map to label
    predicted_label = label_dictionary[predicted_class]

    # Return response
    response = {
        "predicted_character": predicted_label,
        "confidence": confidence
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
