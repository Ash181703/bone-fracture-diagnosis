import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model (update path if needed)
model = tf.keras.models.load_model("bone_fracture_model.h5")

# Fracture classes and healing times
FRACTURE_CLASSES = [
    "Avulsion Fracture", "Comminuted Fracture", "Compression-Crush Fracture",
    "Fracture Dislocation", "GreenStick Fracture", "HairLine Fracture",
    "Impact Fracture", "Intra-articular Fracture", "Null (No Fracture)",
    "Oblique Fracture", "Spiral Fracture"
]

HEALING_TIMES = {
    "Avulsion Fracture": "4-8 weeks (may require surgery)",
    "Comminuted Fracture": "3-6 months (often requires surgery)",
    # ... (include all your healing time entries)
}

def predict(image):
    image = Image.fromarray(image).convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)[0]
    class_idx = np.argmax(preds)
    
    return {
        "Diagnosis": FRACTURE_CLASSES[class_idx],
        "Confidence": f"{preds[class_idx]*100:.2f}%",
        "Healing Time": HEALING_TIMES.get(FRACTURE_CLASSES[class_idx], "N/A")
    }

# Create interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload X-ray"),
    outputs=gr.JSON(label="Results"),
    title="🦴 Bone Fracture Classifier",
    description="Upload an X-ray image to diagnose fracture type",
    examples=[["magical-fantasy-landscape.jpg"], ["sample_xray2.jpg"]]  # Optional example images
)

# Launch
if __name__ == "__main__":
    iface.launch(server_port=1000, server_name="0.0.0.0")