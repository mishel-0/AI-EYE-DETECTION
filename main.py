import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the pretrained AI model
MODEL_PATH = "eye_disease_model.h5"  # Change this if needed
model = load_model(MODEL_PATH)

# Class labels (Modify according to your dataset)
CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

def preprocess_image(image_path):
    """ Load and preprocess an image for model prediction """
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to match model input shape
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_eye_disease(image_path):
    """ Predicts the disease in an eye image """
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    return predicted_class, confidence

# Load an image for testing
image_path = "eye_sample.jpg"  # Replace with your image path
predicted_class, confidence = predict_eye_disease(image_path)

print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}%")

# Display the image
image = cv2.imread(image_path)
cv2.imshow("Eye Scan", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
