from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("bone_fracture_model.h5")

# Define your fracture classes (MUST match model's output order!)
FRACTURE_CLASSES = [
    "Avulsion Fracture",
    "Comminuted Fracture",
    "Compression-Crush Fracture",
    "Fracture Dislocation",
    "GreenStick Fracture",
    "HairLine Fracture",
    "Impact Fracture",
    "Intra-articular Fracture",
    "Null (No Fracture)",
    "Oblique Fracture",
    "Spiral Fracture"
]

# Estimated healing times (rough averages, can be customized)
HEALING_TIME = {
    "Avulsion Fracture": "6-8 weeks",
    "Comminuted Fracture": "12-16 weeks",
    "Compression-Crush Fracture": "8-12 weeks",
    "Fracture Dislocation": "12-20 weeks",
    "GreenStick Fracture": "4-6 weeks",
    "HairLine Fracture": "6-8 weeks",
    "Impact Fracture": "6-10 weeks",
    "Intra-articular Fracture": "12-24 weeks",
    "Null (No Fracture)": "N/A",
    "Oblique Fracture": "8-12 weeks",
    "Spiral Fracture": "10-14 weeks"
}

app = FastAPI()

# Allow CORS (for frontend testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Uploaded file must be an image!")

    try:
        # Read image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = image.resize((224, 224))  # Adjust to your model's input size
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get model prediction
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])

        # Format results
        diagnosis = FRACTURE_CLASSES[predicted_class]
        confidence_percent = round(confidence * 100, 2)
        healing_time = HEALING_TIME.get(diagnosis, "Unknown")

        # Detailed probabilities for all classes
        details = {
            cls: f"{round(prob * 100, 2)}%"
            for cls, prob in zip(FRACTURE_CLASSES, prediction)
        }

        return {
            "diagnosis": f"{diagnosis} (Confidence: {confidence_percent}%)",
            "confidence": confidence,
            "healing_time": healing_time,
            "details": details
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1000)
