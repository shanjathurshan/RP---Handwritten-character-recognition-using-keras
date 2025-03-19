import os
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import cv2

# 🔥 Load Model (Ensure 'my_model.h5' is in the same directory)
MODEL_PATH = "my_model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file '{MODEL_PATH}' not found! Please download it.")
    exit()

print("✅ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Image Preprocessing Function
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to match model input
    img = cv2.bitwise_not(img)  # Invert colors if needed
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Label dictionary (mapping output class to characters)
label_dictionary = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
    45: 'r', 46: 't'
}

@app.route('/')
def main():
    return 'Hello, World!'

# ✅ API Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = preprocess_image(file)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[0][predicted_class]) * 100  # Convert to %

    # Get the corresponding character for the predicted class
    predicted_character = label_dictionary.get(predicted_class, "Unknown")

    response = {
        "predicted_character_digit": int(predicted_class),
        "predicted_character": predicted_character,
        "confidence": confidence
    }

    return jsonify(response)


# ✅ Run Flask Locally
if __name__ == "__main__":
    app.run(debug=True)
