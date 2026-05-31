# 🤖 MiniGPT — Custom GPT-2 Built from Scratch

> A fully custom GPT-2 (124M) implementation in PyTorch — from raw architecture to real-time streaming API. No OpenAI SDK. No shortcuts.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom%20GPT--2-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![HuggingFace](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-yellow?style=flat-square)

---

## 🧠 What is this?

Most people call the OpenAI API. This project **builds the brain itself.**

MiniGPT implements the complete GPT-2 Small (124M) architecture from scratch using PyTorch — Multi-Head Self-Attention, Transformer Blocks, Layer Normalization — without relying on any pre-built LLM library. Official GPT-2 weights are automatically loaded into the custom architecture on first run, and the model is served via a real-time streaming web interface and an OpenAI-compatible REST API.

---

## ✨ Features

- **Custom GPT-2 Architecture** — Every layer hand-coded in PyTorch from the original paper
- **Auto Weight Loading** — Downloads and maps official OpenAI GPT-2 weights on first run
- **Real-Time SSE Streaming** — Tokens stream to the browser word-by-word as they're generated
- **OpenAI-Compatible REST API** — `/v1/completions` endpoint, drop into any existing project
- **Docker + Hugging Face Spaces** — 24/7 free cloud deployment out of the box
- **Custom Training Loop** — Train on your own dataset from scratch if you want

---

## 🏗️ Model Architecture

This project implements the exact **GPT-2 Small (124M)** specification from the original paper:

| Parameter | Value |
|---|---|
| Parameter Count | 124 Million |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Embedding Dimension | 768 |
| Context Window | 1024 tokens |
| Activation Function | GELU |

---

## ⚙️ Prerequisites & System Requirements

- **Python:** 3.9+
- **RAM:** 4GB minimum (8GB recommended)
- **GPU:** Optional — CUDA-compatible NVIDIA GPU is strongly recommended for training. Inference runs on CPU.
- **Dependencies:** `torch`, `flask`, `tiktoken`, `requests`, `tensorflow-cpu`

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Manindra-babu/MiniGpt.git
cd MiniGpt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python app.py
```

On first run, the script automatically downloads the official 124M GPT-2 weights from OpenAI's public servers and maps them tensor-by-tensor into the custom PyTorch architecture. No manual download needed.

### 4. Open the Web UI

Navigate to `http://localhost:5000` in your browser. Type a prompt or click a preset suggestion chip — the response streams onto the screen character-by-character in real time via Server-Sent Events.

---

## 🔌 API Usage

### REST Endpoint (OpenAI-Compatible)

The `/v1/completions` endpoint mirrors OpenAI's API structure exactly, so you can plug MiniGPT into any app that already supports OpenAI.

**cURL:**
```bash
curl -X POST http://localhost:5000/v1/completions \
  -H "Authorization: Bearer sk-custom-llm-token" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "To be or not to be",
    "max_tokens": 100
  }'
```

**Python (drop-in OpenAI replacement):**
```python
import requests

response = requests.post(
    "http://localhost:5000/v1/completions",
    headers={
        "Authorization": "Bearer sk-custom-llm-token",
        "Content-Type": "application/json"
    },
    json={
        "prompt": "What is the meaning of life?",
        "max_tokens": 150
    }
)

print(response.json()["choices"][0]["text"])
```

---

## 🌍 Environment Variables

For cloud deployments (Hugging Face Spaces, Render, etc.):

| Variable | Description | Default |
|---|---|---|
| `API_TOKEN` | Bearer token for API authentication | `sk-custom-llm-token` |
| `PORT` | Port the Flask server binds to | `5000` |

---

## 🧪 Training Your Own Model

The repo includes a full custom training script. You don't need to use the pre-loaded OpenAI weights — swap in your own.

| Setting | Value |
|---|---|
| Dataset | Tiny Shakespeare (or any `.txt` file) |
| Hardware | CPU or NVIDIA GPU (auto-detected via CUDA) |
| Epochs | 4 |
| Batch Size | 4 |
| Max Context Length | 256 tokens |
| Stride | 128 tokens |

```bash
python gpt_archticture.py
```

The final web app uses pre-trained weights by default, but you can point `app.py` at your own checkpoint file after training.

---

## 📁 Project Structure

```
MiniGpt/
├── gpt_archticture.py     # GPT-2 model + training loop
├── app.py                 # Flask server (SSE streaming + REST API)
├── templates/
│   └── index.html         # Chat web UI
├── requirements.txt
└── Dockerfile
```

---

## 🌐 Deployment

Deployed on **Hugging Face Spaces** with Docker for free 24/7 hosting.

Live demo: [huggingface.co/spaces/Manindra-babu/MiniGpt](https://huggingface.co/spaces/Manindra-babu/MiniGpt)

---

## ⚠️ Known Limitations

- **Model size:** 124M parameters. This is not GPT-4. Don't expect complex reasoning or instruction-following.
- **No instruction tuning:** The model uses raw completion weights. The backend wraps prompts in a `Question: ...\nAnswer:` format to encourage Q&A behavior, but responses may still drift.
- **CPU inference speed:** On free-tier hardware (no GPU), generation takes a few seconds per response. The SSE streaming UI masks most of this latency.
- **Context window:** Hard capped at 1024 tokens — very long conversations will get truncated.

---

## 🤝 Contributing

PRs welcome! If you'd like to improve the training loop, add top-p / top-k sampling, or enhance the UI, please open an issue first to discuss your idea.

---

## 📄 License

MIT License — use it, fork it, build on it.

---

## 🙌 Acknowledgements

- [GPT-2 Paper — Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://openai.com/research/language-unsupervised)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for the architecture implementation
- [Hugging Face Spaces](https://huggingface.co/spaces) — free GPU/CPU hosting
