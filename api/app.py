import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess, decode_predictions as efficientnet_decode
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for validation
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Model configuration
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

# Load models dynamically
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
default_model_name = list(models.keys())[0]  # Default to the first model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded!", models=models)
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected!", models=models)
        
        # Validate file type
        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type! Only image files are allowed.", models=models)

        # Save the uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get selected model
            selected_model_name = request.form.get("model", default_model_name)
            model_info = models.get(selected_model_name, models[default_model_name])

            # Load and preprocess the image
            img = load_img(filepath, target_size=model_info["input_size"])
            img_array = img_to_array(img)
            img_array = model_info["preprocess"](img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict and decode results
            model = model_info["model"]
            predictions = model.predict(img_array)
            decoded_predictions = model_info["decode"](predictions, top=5)[0]

            # Format predictions
            results = [
                {"label": label, "probability": f"{prob:.2%}"}
                for (_, label, prob) in decoded_predictions
            ]

            return render_template(
                "index.html", 
                results=results, 
                uploaded_image=filepath, 
                model=selected_model_name, 
                models=models
            )
        
        except Exception as e:
            return render_template("index.html", error=f"Error processing image: {e}", models=models)
    
    return render_template("index.html", models=models)

if __name__ == "__main__":
    app.run(debug=True)