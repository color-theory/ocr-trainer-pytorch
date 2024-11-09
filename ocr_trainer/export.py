import torch
import json
from model import OCRModel

with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

model = OCRModel(num_classes=len(label_mapping))
model.load_state_dict(torch.load("model_ocr_weights.pth", weights_only=True))
model.eval()

dummy_input = torch.randn(1, 1, 50, 50)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model_ocr.onnx",
    export_params=True,
    opset_version=11,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print("Model exported to ONNX format as model_ocr.onnx")