from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import io
import base64
import gc
app = Flask(__name__)

# Global variables for model and processor
model = None
processor = None
device = None

def load_model():
    """Load the model and processor once at startup"""
    global model, processor, device
    
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=None,
        trust_remote_code=True
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/inference', methods=['POST'])
def inference():
    """Handle inference requests"""
    try:
        # Get the uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get the prompt
        prompt = request.form.get('prompt', 'Describe this image.')
        
        # Load and process the image
        image = Image.open(image_file.stream).convert('RGB')
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300)
        
        # Decode only the generated tokens
        generated_ids = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]
        
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return jsonify({
            'success': True,
            'result': result,
            'prompt': prompt
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # 🔽 ADD THIS BLOCK TO FREE VRAM/RAM 🔽
        del image, inputs, messages
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()

@app.route('/text-only', methods=['POST'])
def text_only():
    """Handle text-only inference requests"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=128)  # safer limit

        # Decode only the generated tokens
        generated_ids = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 🔽 Free memory explicitly
        del inputs, outputs, generated_ids, messages, image
        torch.mps.empty_cache()
        gc.collect()
        return jsonify({
            'success': True,
            'result': result,
            'prompt': prompt
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # 🔽 ADD THIS BLOCK TO FREE VRAM/RAM 🔽
        del image, inputs, messages
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5050)