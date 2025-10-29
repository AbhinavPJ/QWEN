from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import io
import base64
import gc
app = Flask(__name__, template_folder='.')
model = None
processor = None
device = None
def load_model():
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
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        prompt = request.form.get('prompt', 'Describe this image.')
        image = Image.open(image_file.stream).convert('RGB')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300)
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
        # Free up the memory
        del image, inputs, messages
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
@app.route('/text-only', methods=['POST'])
def text_only():
    inputs = None
    messages = None
    outputs = None
    generated_ids = None
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=128)

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
        # Safe cleanup - delete variables only if they exist
        if inputs is not None:
            del inputs
        if outputs is not None:
            del outputs
        if generated_ids is not None:
            del generated_ids
        if messages is not None:
            del messages
            
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()


if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=7860)