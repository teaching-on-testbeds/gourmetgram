import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import tritonclient.http as httpclient # New: for making requests to Triton

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

TRITON_SERVER_URL=os.environ['TRITON_SERVER_URL']
FOOD11_MODEL_NAME=os.environ['FOOD11_MODEL_NAME']

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0).numpy()

# New! for making requests to Triton
def request_triton(image_tensor):
    try:
        client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

        input_data = httpclient.InferInput("input", image_tensor.shape, "FP32")
        input_data.set_data_from_numpy(image_tensor)

        response = client.infer(FOOD11_MODEL_NAME, [input_data])
        output = response.as_numpy("output")

        classes = np.array([
            "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
            "Vegetable/Fruit"
        ])

        predicted_class_idx = np.argmax(output)
        predicted_class = classes[predicted_class_idx]
        probability = float(output[0][predicted_class_idx])

        return predicted_class, probability

    except Exception as e:
        return str(e), 0.0

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        # New! using request_triton
        img_path = "./instance/uploads/" + secure_filename(f.filename)
        img = Image.open(img_path).convert('RGB')  
        img_tensor = preprocess_image(img)
        preds, probs = request_triton(img_tensor)
        return '<button type="button" class="btn btn-info btn-sm">' + str(preds) + '</button>' 
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/test', methods=['GET'])
def test():
    img_path = "./instance/uploads/test_image.jpeg"
    img = Image.open(img_path).convert('RGB')  
    img_tensor = preprocess_image(img)
    preds, probs = request_triton(img_tensor)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
