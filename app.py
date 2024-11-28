import io
import base64
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input as efficientnet_preprocess,
    decode_predictions as efficientnet_decode,
)
from tensorflow.keras.applications.resnet import (
    ResNet50,
    preprocess_input as resnet_preprocess,
    decode_predictions as resnet_decode,
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

MODEL_CONFIG = {
    "EfficientNetB0": {
        "class": EfficientNetB0,
        "weights": "imagenet",
        "preprocess": efficientnet_preprocess,
        "decode": efficientnet_decode,
        "description": "Optimized for performance and efficiency.",
        "input_size": (224, 224),
    },
    "ResNet50": {
        "class": ResNet50,
        "weights": "imagenet",
        "preprocess": resnet_preprocess,
        "decode": resnet_decode,
        "description": "Deep residual network for image recognition.",
        "input_size": (224, 224),
    },
}

def load_models(config):
    models = {}
    for model_name, settings in config.items():
        models[model_name] = {
            "model": settings["class"](weights=settings["weights"]),
            "preprocess": settings["preprocess"],
            "decode": settings["decode"],
            "description": settings["description"],
            "input_size": settings["input_size"],
        }
    return models

models = load_models(MODEL_CONFIG)
default_model_name = list(models.keys())[0]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded!", models=models)
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected!", models=models)
        
        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type!", models=models)
        
        try:
            selected_model_name = request.form.get("model", default_model_name)
            model_info = models.get(selected_model_name, models[default_model_name])

            file_bytes = io.BytesIO(file.read())
            img = load_img(file_bytes, target_size=model_info["input_size"])
            img_array = img_to_array(img)
            img_array = model_info["preprocess"](img_array)
            img_array = np.expand_dims(img_array, axis=0)

            model = model_info["model"]
            predictions = model.predict(img_array)
            decoded_predictions = model_info["decode"](predictions, top=5)[0]

            results = [
                {"label": label, "probability": f"{prob:.2%}"}
                for (_, label, prob) in decoded_predictions
            ]

            # Encode the uploaded image as base64
            file_bytes.seek(0)
            image = Image.open(file_bytes)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{encoded_image}"

            return render_template(
                "index.html",
                results=results,
                uploaded_image=image_data_url,  # Pass the base64 data URL
                model=selected_model_name,
                models=models,
            )
        except Exception as e:
            return render_template("index.html", error=f"Error processing image: {e}", models=models)

    return render_template("index.html", models=models)

if __name__ == "__main__":
    app.run(debug=True)
