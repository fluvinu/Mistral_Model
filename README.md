# 🤖 Mistral-7B Instruct Chatbot (GGUF + Google Colab + Gradio)

This project creates a simple web-based chatbot UI using the [Mistral-7B-Instruct-v0.1](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) model in **GGUF format**, running locally with `llama-cpp-python` inside **Google Colab**, and served to the web using **Gradio** with a public share link.

---

## 🚀 Features

- 💬 Chat with Mistral 7B Instruct using natural language
- ⚙️ Runs fully offline in Google Colab
- 🌐 Accessible via a public Gradio web link
- 🧠 Powered by quantized GGUF model via `llama-cpp-python`
- 🪶 Lightweight and fast (Q4_K_M quantization)

---

## 📦 Dependencies

This runs in Google Colab and requires the following Python libraries:

- `llama-cpp-python`
- `huggingface-hub`
- `gradio`

All are auto-installed in the notebook.

---

## 🔧 How to Run

1. Open the Colab notebook:  
   👉 [**Open in Colab**](https://colab.research.google.com/)

2. Paste and run the following code block in the Colab cell:

```python
# Install libraries
!pip install -q llama-cpp-python huggingface-hub gradio

# Download model
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)

# Load model
from llama_cpp import Llama
llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8, verbose=False)

# Chat function
def chat_with_mistral(prompt):
    formatted = f"[INST] {prompt.strip()} [/INST]"
    out = llm.create_completion(prompt=formatted, max_tokens=256, temperature=0.7)
    return out["choices"][0]["text"].strip()

# Gradio UI
import gradio as gr
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chat with Mistral 7B Instruct (GGUF Model in Colab)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message", placeholder="Ask anything...", lines=1)
    def respond(user_message, chat_history):
        bot_reply = chat_with_mistral(user_message)
        chat_history.append((user_message, bot_reply))
        return "", chat_history
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    gr.Button("Clear").click(lambda: [], None, chatbot, queue=False)
demo.launch(share=True)
