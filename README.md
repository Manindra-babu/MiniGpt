# Custom GPT-2 Architecture & Streaming Web App 🚀

This repository contains a PyTorch implementation of a GPT-2 style Large Language Model (LLM) built from scratch, complete with a full-stack web application featuring real-time Server-Sent Events (SSE) streaming and an OpenAI-compatible REST API.

## Features
- **Custom PyTorch Architecture**: Implementation of `TransformerBlock`, `MultiHeadAttention`, and `LayerNorm` from the ground up.
- **OpenAI Compatible API**: Features a `/v1/completions` REST endpoint with Bearer Token authentication, matching the official OpenAI API schema.
- **Real-Time Streaming**: Custom frontend UI (HTML/CSS/JS) using SSE to stream generated tokens to the client instantaneously.
- **Pre-trained Weights Loading**: Automatically downloads and maps official 124M OpenAI GPT-2 weights into the custom PyTorch model for high-quality text generation.
- **Dockerized**: Ready for immediate cloud deployment on platforms like Hugging Face Spaces.

## Tech Stack
- **AI/ML Backend**: PyTorch, Tiktoken
- **Web Server**: Flask (Python)
- **Frontend**: Vanilla Javascript, CSS3, HTML5
- **Deployment**: Docker

## Getting Started

### Local Setup
1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. The server will automatically download the 124M weights (if not already present) and start the web UI on `http://localhost:5000`.

### API Usage
You can programmatically interact with the model using standard HTTP requests:

```bash
curl -X POST "http://localhost:5000/v1/completions" \
     -H "Authorization: Bearer sk-custom-llm-token" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "Question: What is the meaning of life?\nAnswer:",
         "max_tokens": 150
     }'
```

## Architecture Details
The custom GPT-2 architecture is defined in `gpt_model.py` and strictly follows the original paper's specifications:
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **Context Length**: 1024
