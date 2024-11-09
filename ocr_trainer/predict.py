import torch
import json
from flask import Flask, request, jsonify
from model import OCRModel

with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

model = OCRModel(num_classes=len(label_mapping))
model.load_state_dict(torch.load('model_ocr_weights.pth', weights_only=True))
model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    images = data.get('images')

    image_tensors = [torch.tensor(img, dtype=torch.float32).view(1, 50, 50) for img in images]
    image_batch = torch.stack(image_tensors)
    with torch.no_grad():
        outputs = model(image_batch)
        predictions = torch.argmax(outputs, dim=1).tolist()

    predicted_labels = [label_mapping[idx] for idx in predictions]
    result = ''.join(predicted_labels)  # Combine into a string

    return jsonify({'prediction': result})

app.run(debug=True)
