from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Create a directory for uploads if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# def prepare_image(image_path):
#     img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
#     img = img.resize((150, 150))  # Resize image to match model input
#     img_array = img_to_array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array


def prepare_image(image_path):
    """Load and preprocess the image for prediction."""
    img = load_img(image_path, target_size=(150, 150))  # Resize image to match model input
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Define class labels (make sure this matches your model's labels)
    class_labels = {0: 'Tomato___Bacterial_spot', 1: 'Tomato___Early_blight',
                    2: 'Tomato___Late_blight', 3: 'Tomato___Leaf_Mold',
                    4: 'Tomato___Septoria_leaf_spot', 5: 'Tomato___Spider_mites Two-spotted_spider_mite',
                    6: 'Tomato___Target_Spot', 7: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    8: 'Tomato___Tomato_mosaic_virus', 9: 'Tomato___healthy'}

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)


        def predict(image_path):
            """Predict the class of an image."""
            img_array = prepare_image(image_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            class_name = class_labels.get(predicted_class, 'Unknown')
            return class_name, confidence

        class_name, confidence = predict(filepath)

        # Clean up: remove the saved file after prediction
        os.remove(filepath)

        return jsonify({
            'class_name': class_name,
            'confidence': float(confidence)
        })


if __name__ == '__main__':
    app.run(debug=True)
