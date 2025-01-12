import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Step 1: Load the pre-trained model and tokenizer
model_name = "roberta-base"
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Step 2: Create dummy input for the model
dummy_input = tokenizer("This is a sample input", return_tensors="pt")

# Step 3: Export the model to ONNX format
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    #"webapp/roberta-sequence-classification-9.onnx", #comment out to build using Dockerfile 
    "roberta-sequence-classification-9.onnx", 
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    #opset_version=9,
    opset_version=14,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: 'sequence'},
        "attention_mask": {0: "batch_size", 1: 'sequence'},
        "logits": {0: "batch_size"}
    }
)

print("Model has been converted to ONNX format and saved as 'webapp/roberta-sequence-classification-9.onnx'")
