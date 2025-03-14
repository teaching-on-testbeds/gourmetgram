import numpy as np
import requests
import tritonclient.http as httpclient
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

TRITON_SERVER_URL = os.environ['TRITON_SERVER_URL']
FOOD11_MODEL_NAME = os.environ['FOOD11_MODEL_NAME']

# Class labels
FOOD_CLASSES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
    "Vegetable/Fruit"
]

# Image preprocessing - now in client
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).numpy()  # onnx model needs numpy array
    return np.expand_dims(img_tensor, axis=0).astype(np.float32)  # and needs a batch dimension

# New! Request to Triton after preprocessing
def request_triton(image_path):
    try:
        # Connect to Triton server
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

        # Preprocess image
        processed_image = preprocess_image(image_path)

        # Prepare inputs and outputs for Triton
        inputs = [httpclient.InferInput("input", processed_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(processed_image)

        outputs = [httpclient.InferRequestedOutput("output")]

        # Run inference
        results = triton_client.infer(model_name=FOOD11_MODEL_NAME, inputs=inputs, outputs=outputs)

        # Get the softmax output
        softmax_probs = results.as_numpy("output")[0]  # Shape: (11,)

        # Convert softmax output to class label
        predicted_index = np.argmax(softmax_probs)
        predicted_label = FOOD_CLASSES[predicted_index]
        probability = float(softmax_probs[predicted_index])

        return predicted_label, probability

    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        img_path = "./instance/uploads/" + secure_filename(f.filename)
        preds, probs = request_triton(img_path)
        if preds:
            return f'<button type="button" class="btn btn-info btn-sm">{preds} ({probs:.2f})</button>'
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/test', methods=['GET'])
def test():
    img_path = "./instance/uploads/test_image.jpeg"
    preds, probs = request_triton(img_path)
    return f"{preds} ({probs:.2f})"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)