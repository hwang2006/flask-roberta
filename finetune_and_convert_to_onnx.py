import os
import torch
import argparse
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def train_model(data_dir, output_dir, onnx_output_dir, logging_dir, model_name="roberta-base", num_train_epochs=3):
    # Load the pre-trained model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Load a sentiment analysis dataset
    dataset = load_dataset("imdb", cache_dir=data_dir)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Prepare the dataset for training
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    # Define training arguments with logging enabled
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Ensure save_strategy matches evaluation_strategy
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=logging_dir,  # Directory for storing logs
        logging_strategy="steps",  # Log metrics every few steps
        logging_steps=10,  # Log every 10 steps
        resume_from_checkpoint=True,  # Enable resuming from checkpoint
    )

    # Get the latest checkpoint if it exists
    def get_latest_checkpoint(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]), reverse=True)
        return checkpoints[0] if checkpoints else None

    # Create a Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Check if a checkpoint exists and resume from it
    latest_checkpoint = get_latest_checkpoint(training_args.output_dir)

    # Fine-tune the model, resuming from the checkpoint if it exists
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # Save the fine-tuned model
    model.save_pretrained("fine-tuned-roberta")

    # Create the directory for the ONNX model if it does not exist
    os.makedirs(onnx_output_dir, exist_ok=True)

    # Export the fine-tuned model to ONNX format
    dummy_input = tokenizer("This is a sample input", return_tensors="pt")

    # Move model and dummy input to the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

    onnx_output_path = os.path.join(onnx_output_dir, "roberta-sequence-classification-9-finetuned.onnx")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: 'sequence'},
            "attention_mask": {0: "batch_size", 1: 'sequence'},
            "logits": {0: "batch_size"}
        }
    )

    print(f"Fine-tuned model has been converted to ONNX format and saved as '{onnx_output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Roberta model and export it to ONNX format")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to cache the dataset")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save training outputs and checkpoints")
    parser.add_argument("--onnx_output_dir", type=str, default="fine-tuned-roberta/", help="Directory to save the exported ONNX model")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory to store logs")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pre-trained model name")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        onnx_output_dir=args.onnx_output_dir,
        logging_dir=args.logging_dir,
        model_name=args.model_name,
        num_train_epochs=args.num_train_epochs
    )