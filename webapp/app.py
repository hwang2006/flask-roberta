from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime

app = Flask(__name__)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")
#session = onnxruntime.InferenceSession("roberta-sequence-classification-9-finetuned.onnx")

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

@app.route("/")
def home():
    return "<h2>RoBERTa sentiment analysis</h2>"

@app.route("/predict", methods=["POST"])
def predict():
    # Tokenize the input text
    tokens = tokenizer.encode_plus(
        request.json[0],
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    inputs = {
        session.get_inputs()[0].name: to_numpy(input_ids),
        session.get_inputs()[1].name: to_numpy(attention_mask)
    }

    # Debug: print the inputs
    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)

    out = session.run(None, inputs)

    # Debug: print the raw output from the model
    print("Model output:", out)

    result = np.argmax(out)
    # Debug: print the result
    print("Predicted class:", result)

    return jsonify({"positive": bool(result)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
