from flask import Flask, request, jsonify
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import os

app = Flask(__name__)

# Load your Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "your-api-key-here"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Setup the model
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    generation_config=GenerationConfig(
        temperature=0.7,
        max_output_tokens=300,
        top_p=1.0,
        top_k=40
    ),
    system_instruction="You are a friendly assistant who helps users clearly and kindly."
)

@app.route('/api/gemini-chat', methods=['POST'])
def gemini_chat():
    data = request.get_json()
    messages = data.get("messages", [])

    chat = model.start_chat(history=[
        {"role": msg["role"], "parts": [msg["content"]]} for msg in messages
    ])

    if messages and messages[-1]["role"] == "user":
        response = chat.send_message(messages[-1]["content"])
        return jsonify({"reply": response.text})
    else:
        return jsonify({"reply": "No user input received."})

if __name__ == '__main__':
    app.run(debug=True)