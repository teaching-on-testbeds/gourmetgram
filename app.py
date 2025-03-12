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

# New! for making requests to Triton
def request_triton(image_path):
    try:
        # Connect to Triton server
        client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

        # Prepare inputs and outputs
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        inputs = []
        inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))

        encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
        input_data = np.array([[encoded_str]], dtype=object)
        inputs[0].set_data_from_numpy(input_data)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput("FOOD_LABEL", binary_data=False))
        outputs.append(httpclient.InferRequestedOutput("PROBABILITY", binary_data=False))

        # Run inference
        results = triton_client.infer(model_name=FOOD11_MODEL_NAME, inputs=inputs, outputs=outputs)

        predicted_class = results.as_numpy("FOOD_LABEL")
        probability = results.as_numpy("PROBABILITY")

        return predicted_class, probability

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
        # Get the file from post request
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        # New! using request_triton
        img_path = "./instance/uploads/" + secure_filename(f.filename)
        preds, probs = request_triton(img_path)
        if preds:
            return '<button type="button" class="btn btn-info btn-sm">' + str(preds) + '</button>' 
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/test', methods=['GET'])
def test():
    img_path = "./instance/uploads/test_image.jpeg"
    preds, probs = request_triton(img_path)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
