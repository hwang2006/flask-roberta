# Flask ONNX Model Inference API

This repository contains a Flask application for performing sentiment analysis using an ONNX model fine-tuned with `Roberta`. It accepts text input and predicts whether the sentiment is positive or not.

---

## Features
- Provides a RESTful API to interact with an ONNX model.
- Tokenizes input text using Hugging Face's `RobertaTokenizer`.
- Lightweight Flask application for easy deployment.
- Returns predictions in JSON format.

---

## Requirements

### System Requirements
- Python 3.8+
- pip

### Python Dependencies
Install the required Python libraries:
```bash
pip install flask transformers torch numpy onnxruntime
```

---

## Usage

### 1. Start the Flask Server
Run the following command to start the server:
```bash
python webapp/app.py
```
By default, the server will run on `http://127.0.0.1:5000/`.

### 2. Test the API

#### Root Endpoint:
```bash
curl http://127.0.0.1:5000/
```
**Response:**
```html
<h2>RoBERTa sentiment analysis</h2>
```

#### Prediction Endpoint:
Make a POST request to the `/predict` endpoint with a JSON array containing a string phrase.

##### Example Request:
```bash
curl -X POST "http://127.0.0.1:5000/predict" \
-H "Content-Type: application/json" \
-d '["This is a fantastic example!"]'
```

##### Example Response:
```json
{
  "positive": true
}
```

---

## Deployment

### Local Deployment
1. Clone the repository.
2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask application:
   ```bash
   python app.py
   ```

### Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t flask-onnx-app .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 flask-onnx-app
   ```

---

## Debugging

### Debugging Inputs
- Input tensors (`input_ids` and `attention_mask`) are logged to the console for inspection.

### Debugging Outputs
- Raw model outputs and the predicted class are logged to the console for debugging purposes.

---

## Notes
- Ensure the ONNX model file is placed in the directory specified in `app.py`.
- Update the model path in `app.py` if needed:
  ```python
  session = onnxruntime.InferenceSession("path/to/your/onnx/model.onnx")
  ```



