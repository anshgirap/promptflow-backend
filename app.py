from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Hugging Face Router Config

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")


HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Prompt Template

TEMPLATE = """
You are a professional prompt engineer for AI image generation models.

Transform the user idea into a precise, high-impact, image-ready prompt.

Target platform: {platform}

You MUST output exactly 5 lines.
Each line must be under 20 words.
Each line must add a new visual layer.

If you break these rules, regenerate internally and fix it before responding.

Line structure:
1) Subject
2) Environment
3) Lighting
4) Color & mood
5) Camera & depth

No explanations. No commentary.

User idea:
{user_prompt}
"""

# Hugging Face Call

def call_hf(prompt):
    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 400
    }

    for _ in range(5):
        try:
            r = requests.post(HF_URL, headers=HEADERS, json=payload, timeout=60)
            data = r.json()
        except Exception as e:
            return None, str(e)

        # HF router error
        if "error" in data:
            msg = data["error"].get("message", "")
            if "loading" in msg.lower():
                time.sleep(5)
                continue
            return None, msg

        try:
            return data["choices"][0]["message"]["content"], None
        except:
            return None, f"Bad response: {data}"

    return None, "Model did not load in time"

# API Endpoint

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_prompt = data.get("prompt", "")
    platform = data.get("platform", "Generic")

    if not user_prompt.strip():
        return jsonify({"error": "Prompt is empty"}), 400

    final_prompt = TEMPLATE.format(
        platform=platform,
        user_prompt=user_prompt
    )

    result, error = call_hf(final_prompt)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({
        "expanded_prompt": result.strip()
    })

# Run

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)