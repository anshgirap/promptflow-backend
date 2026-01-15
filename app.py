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
You are a professional cinematic art director and prompt engineer whose only task is to transform a short idea into a powerful, high-quality image generation prompt.

Your job is NOT to explain, comment, list, number, summarize, or analyze.
Your job is to OUTPUT ONE SINGLE VISUAL PROMPT.

The prompt must feel like a professional art brief written for an elite digital artist.

Target platform: {platform}

The output MUST follow ALL of these rules with zero exceptions:

• Write in flowing descriptive paragraphs — never use bullet points, numbers, dashes, or lists  
• Do NOT include explanations, disclaimers, or commentary  
• Do NOT reference the user, rules, platforms, or instructions  
• Do NOT include quotation marks  
• Do NOT include headings  
• Do NOT include any meta-language  
• Output ONLY the final prompt  

The prompt must be between 4 and 6 full lines of rich cinematic description.

The scene must always include:
A clear primary subject  
A defined environment or setting  
Lighting direction and quality  
A dominant color mood or palette  
Depth, perspective, or camera framing  

The visual must feel spatially grounded — the subject must feel placed inside a real scene, not floating in empty space.

The writing style must be:
Cinematic  
Visually dense  
Emotionally atmospheric  
Precise but not technical  
Vivid without being verbose  

Avoid generic phrases like “beautiful,” “stunning,” or “nice.”  
Use concrete visual language, texture, light, atmosphere, and mood instead.

The final result should read like a scene being described to a world-class concept artist.

Now generate the image prompt based on this idea:

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