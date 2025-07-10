from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from typing import Dict, Any

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

# Healing time information for each fracture type
HEALING_TIMES = {
    "Avulsion Fracture": "4-8 weeks (may require surgery if tendon is involved)",
    "Comminuted Fracture": "3-6 months (often requires surgical intervention)",
    "Compression-Crush Fracture": "8-12 weeks (longer for spinal fractures)",
    "Fracture Dislocation": "6-12 weeks (requires reduction and often surgery)",
    "GreenStick Fracture": "3-6 weeks (common in children, heals faster)",
    "HairLine Fracture": "4-6 weeks (typically heals with immobilization)",
    "Impact Fracture": "6-10 weeks (depends on bone and impact severity)",
    "Intra-articular Fracture": "8-16 weeks (may require surgery, longer recovery)",
    "Null (No Fracture)": "No healing time needed",
    "Oblique Fracture": "6-12 weeks (depends on bone and displacement)",
    "Spiral Fracture": "8-12 weeks (often requires immobilization or surgery)"
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
        healing_time = HEALING_TIMES.get(diagnosis, "Varies depending on severity")

        # Detailed probabilities for all classes
        details = {
            cls: {
                "probability": f"{round(prob * 100, 2)}%",
                "healing_time": HEALING_TIMES.get(cls, "N/A")
            }
            for cls, prob in zip(FRACTURE_CLASSES, prediction)
        }

        return {
            "diagnosis": diagnosis,
            "confidence": f"{confidence_percent}%",
            "healing_time": healing_time,
            "details": details
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1000)  # Changed port to 1000