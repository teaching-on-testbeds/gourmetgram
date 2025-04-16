import numpy as np
import requests
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import base64
from mimetypes import guess_type
from datetime import datetime
import boto3
import uuid
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)  # can adjust max_workers as needed

s3 = boto3.client(
    's3',
    endpoint_url=os.environ['MINIO_URL'],  # e.g. 'http://minio:9000'
    aws_access_key_id=os.environ['MINIO_USER'],
    aws_secret_access_key=os.environ['MINIO_PASSWORD'],
    region_name='us-east-1'  # required for the boto client but not used by MinIO
)

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']  # FastAPI server URL

# Helper function for getting object key
def get_object_key(preds, prediction_id, filename):
    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
        "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
        "Vegetable/Fruit"])
    pred_index = np.where(classes == preds)[0][0]
    class_dir = f"class_{pred_index:02d}"
    ext = os.path.splitext(filename)[1]
    return f"{class_dir}/{prediction_id}{ext}"

def upload_production_bucket(img_path, preds, confidence, prediction_id):
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    bucket_name = "production"
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = get_object_key(preds, prediction_id, img_path)
    
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(f, 
            bucket_name, 
            s3_key, 
            ExtraArgs={'ContentType': content_type}
            )

    # tag the object with predicted class and confidence
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=s3_key,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_class', 'Value': preds},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )

# For making requests to FastAPI
def request_fastapi(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}
        
        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", json=payload)
        response.raise_for_status()
        
        result = response.json()
        predicted_class = result.get("prediction")
        probability = result.get("probability")
        
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
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        img_path = os.path.join(app.instance_path, 'uploads', secure_filename(f.filename))

        # create a unique filename for the image
        prediction_id = str(uuid.uuid4())
        
        preds, probs = request_fastapi(img_path)
        if preds:
            executor.submit(upload_production_bucket, img_path, preds, probs, prediction_id) 
            # New! return a flag icon along with the class label
            # Clicking the flag icon submits a form with the object key passed as an argument, 
            # so we can tag it as "flagged"
            s3_key = get_object_key(preds, prediction_id, img_path)
            flag_icon = f'''
                <form method="POST" action="/flag/{s3_key}" style="display:inline">
                    <button type="submit" class="btn btn-outline-warning btn-sm">🚩</button>
                </form>'''
            return f'<button type="button" class="btn btn-info btn-sm">{preds}</button> {flag_icon}'

    return '<a href="#" class="badge badge-warning">Warning</a>'

# New! tag an object as "flagged" if user clicks the "flag" icon
@app.route('/flag/<path:key>', methods=['POST'])
def flag_object(key):
    bucket = "production"
    current_tags = s3.get_object_tagging(Bucket=bucket, Key=key)['TagSet']
    tags = {t['Key']: t['Value'] for t in current_tags}

    if "flagged" not in tags:
        tags["flagged"] = "true"
        tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
        s3.put_object_tagging(Bucket=bucket, Key=key, Tagging={'TagSet': tag_set})

    return '', 204  # No Content: allows staying on the same page  

@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    preds, probs = request_fastapi(img_path)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
