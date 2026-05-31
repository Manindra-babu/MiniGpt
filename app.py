import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from flask import Flask, request, jsonify, send_from_directory, Response
import time
import json
import tiktoken
from gpt_model import GPTModel, GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text, generate, model_configs, load_weights_into_gpt
from gpt_download3 import download_and_load_gpt2

app = Flask(__name__, static_folder='static')

# Configuration
API_TOKEN = os.environ.get("API_TOKEN", "sk-custom-llm-token")

# Initialize tokenizer and device
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up model configuration
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# Load the model
print("Loading model...")
gpt = GPTModel(NEW_CONFIG)

# Attempt to load OpenAI pre-trained weights for the demo
try:
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="GPT")
    load_weights_into_gpt(gpt, params)
    print("Loaded pre-trained weights from OpenAI.")
except Exception as e:
    print(f"Could not load pre-trained weights: {e}. Model will use random initialization.")

gpt.to(device)
gpt.eval()

@app.route('/')
def serve_index():
    # Debugging: show files if index.html is missing
    if os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    elif os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    else:
        # If the file is truly missing, let's list what is actually in the folder to debug!
        files = "\n".join(os.listdir('.'))
        return f"ERROR: Could not find index.html. Here are the files in your folder:<br><pre>{files}</pre>"

@app.route('/style.css')
def serve_css():
    if os.path.exists(os.path.join(app.static_folder, 'style.css')):
        return send_from_directory(app.static_folder, 'style.css')
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def serve_js():
    if os.path.exists(os.path.join(app.static_folder, 'script.js')):
        return send_from_directory(app.static_folder, 'script.js')
    return send_from_directory('.', 'script.js')

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 50)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 50)
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    # Wrap the prompt to encourage the model to answer questions
    is_question = prompt.strip().endswith('?')
    if is_question:
        formatted_prompt = f"Question: {prompt}\nAnswer:"
    else:
        formatted_prompt = prompt

    encoded = text_to_token_ids(formatted_prompt, tokenizer).to(device)
    
    def generate_stream():
        nonlocal encoded
        previous_text = ""
        eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        for _ in range(max_new_tokens):
            idx_cond = encoded[:, -NEW_CONFIG["context_length"]:]
            with torch.no_grad():
                logits = gpt(idx_cond)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(device), logits)
            
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
            if idx_next.item() == eos_id:
                break
                
            encoded = torch.cat((encoded, idx_next), dim=1)
            
            current_text = token_ids_to_text(encoded, tokenizer)
            
            if current_text.startswith(formatted_prompt):
                response_text = current_text[len(formatted_prompt):].lstrip()
            else:
                response_text = current_text
                
            new_text = response_text[len(previous_text):]
            if new_text:
                yield f"data: {json.dumps({'text': new_text})}\n\n"
                previous_text = response_text

    return Response(generate_stream(), mimetype='text/event-stream')

@app.route('/v1/completions', methods=['POST'])
def v1_completions():
    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_TOKEN:
        return jsonify({"error": "Unauthorized. Invalid or missing API token."}), 401
    
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = data.get('prompt')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 50)

    try:
        encoded = text_to_token_ids(prompt, tokenizer).to(device)
        token_ids = generate(
            model=gpt,
            idx=encoded,
            max_new_tokens=max_tokens,
            context_size=NEW_CONFIG["context_length"],
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # Format like OpenAI's completion response
        response_payload = {
            "id": "cmpl-" + str(int(time.time())),
            "object": "text_completion",
            "created": int(time.time()),
            "model": "custom-gpt-124m",
            "choices": [
                {
                    "text": decoded_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(encoded[0]),
                "completion_tokens": max_tokens,
                "total_tokens": len(encoded[0]) + max_tokens
            }
        }
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
