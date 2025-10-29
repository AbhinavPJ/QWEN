# Qwen3-VL Flask API Deployment

This project deploys the `Qwen/Qwen3-VL-2B-Instruct` model as a Flask web application, providing a simple web UI and callable API endpoints for both vision-text and text-only inference.

![alt text](https://github.com/AbhinavPJ/QWEN/blob/main/result.png?raw=true)

##  Live Demo

The easiest way to try this project is to use the **live version deployed on Hugging Face Spaces**:

**[https://huggingface.co/spaces/AbhinavPJ/QWEN_tryingout](https://huggingface.co/spaces/AbhinavPJ/QWEN_tryingout)**

You can use the interactive web UI or test the callable API endpoints listed below directly at this URL.

---

## API Endpoints

The server exposes two primary API endpoints. You can call these on the live demo URL or on your local instance.

### 1. Vision + Text Inference

* **Endpoint:** `POST /inference`
* **Request:** `multipart/form-data`
    * `image`: The image file (e.g., `image.png`).
    * `prompt`: A string (e.g., "Describe this image.").
* **Response:** `application/json`
    ```json
    {
      "success": true,
      "result": "A description of the image...",
      "prompt": "Describe this image."
    }
    ```

**Example (Python `requests`):**

See `test_vision_api.py`.

```python
import requests

# Use the live demo URL
url = "https://abhinavpj-qwen-tryingout.hf.space/inference"
# Or a local URL
# url = "http://127.0.0.1:7860/inference"

image_path = "image.png"
prompt_text = "What objects are in this image?"

files = {'image': (image_path, open(image_path, 'rb'), 'image/png')}
payload = {'prompt': prompt_text}

try:
    response = requests.post(url, files=files, data=payload)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

### 2. Text-Only Inference

* **Endpoint:** `POST /text-only`
* **Request:** `application/json`
    ```json
    {
      "prompt": "What is the capital of France?"
    }
    ```
* **Response:** `application/json`
    ```json
    {
      "success": true,
      "result": "The capital of France is Paris.",
      "prompt": "What is the capital of France?"
    }
    ```

**Example (Python `requests`):**

See `testing_huggingface.py`.

```python
import requests

# Use the live demo URL
url = "https://abhinavpj-qwen-tryingout.hf.space/text-only"
# Or a local URL
# url = "http://127.0.0.1:7860/text-only"

payload = {'prompt': 'What is the capital of France?'}

try:
    response = requests.post(url, json=payload)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

## Features

* **Interactive Web UI**: A clean, modern frontend with tabs for "Vision + Text" and "Text Only" modes.
* **File Upload**: Supports drag-and-drop and click-to-upload for images.
* **Dockerized**: Includes a Dockerfile for easy and reproducible deployment.
* **Memory Management**: Implements `gc.collect()` and `torch.mps.empty_cache()` for more stable inference.

## Run Locally

If you prefer to run the application on your own machine, you can follow these steps.

### Installation

1. Clone the repository:
   ```bash
   git clone https://huggingface.co/spaces/AbhinavPJ/QWEN_tryingout
   cd QWEN_tryingout
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. The model will be downloaded and loaded into memory (this may take a few minutes).

3. Open your browser and navigate to `http://127.0.0.1:7860`.


## 📁 Project Structure

```
.
├── 🚀 app.py                    # The main Flask server
├── 📄 Dockerfile                # For building the Docker container
├── 🎨 index.html                # The HTML/CSS/JS frontend
├── 📝 requirements.txt          # Python dependencies
├── 🖼️ image.png                 # An example image for testing
├── 🔬 test_vision_api.py        # Test script for the /inference endpoint
└── 🔬 testing_huggingface.py    # Test script for the /text-only endpoint
```

## License

This project is open source and available under the MIT License.

##  Contributing

Contributions, issues, and feature requests are welcome!

##  Acknowledgments

* Built with [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
* Powered by Flask and Hugging Face Transformers