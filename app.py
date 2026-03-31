import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from mimetypes import guess_type
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

model = models.mobilenet_v2(weights=None)
num_ftrs = model.last_channel
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 11)
)
state = torch.load("food11.pth", map_location=torch.device('cpu'))
model.load_state_dict(state)
model.eval()

classes = np.array([
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"
])

GCS_LABELED_BUCKET = os.environ.get("GCS_LABELED_BUCKET")
executor = ThreadPoolExecutor(max_workers=2)
storage_client = storage.Client() if GCS_LABELED_BUCKET else None

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def model_predict(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = preprocess_image(img)

    with torch.no_grad():
        output = model(img)
        prob, predicted_class = torch.max(output, 1)

    return classes[predicted_class.item()], torch.sigmoid(prob).item()

def upload_labeled_bucket(img_path, preds, confidence, prediction_id):
    if not storage_client:
        return

    pred_index = int(np.where(classes == preds)[0][0])
    class_dir = f"class_{pred_index:02d}"
    _, ext = os.path.splitext(img_path)
    if not ext:
        ext = ".jpg"
    object_name = f"{class_dir}/{prediction_id}{ext}"

    content_type = guess_type(img_path)[0] or "application/octet-stream"
    bucket = storage_client.bucket(GCS_LABELED_BUCKET)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(img_path, content_type=content_type)
    blob.metadata = {
        "predicted_class": str(preds),
        "confidence": f"{confidence:.3f}",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    blob.patch()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        img_path = os.path.join(app.instance_path, 'uploads', filename)
        f.save(img_path)

        prediction_id = str(uuid.uuid4())
        preds, probs = model_predict(img_path, model)

        if GCS_LABELED_BUCKET:
            executor.submit(upload_labeled_bucket, img_path, preds, probs, prediction_id)

        return '<button type="button" class="btn btn-info btn-sm">' + str(preds) + '</button>'

    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/test', methods=['GET'])
def test():
    preds, probs = model_predict("./instance/uploads/test_image.jpeg", model)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
