from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load Pretrained Model
MODEL_PATH = "eye_disease_model.h5"  # Replace with your model path
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (Modify according to your dataset)
CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

def preprocess_image(image: Image.Image):
    """ Preprocess image for model prediction. """
    image = image.resize((224, 224))  # Resize to match model input shape
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """ API endpoint to receive an image and return the AI model's prediction. """
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)

        # Model Prediction
        predictions = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100

        return {"prediction": predicted_class, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Eye Disease Detection API is running"}
