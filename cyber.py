from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import os
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="Cyber Chatbot using Groq + LLaMA 3")



# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming requests
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # Optional: [{"role": "user", "content": "..."}, ...]

# Set your Groq credentials and parameters
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

# System prompt
system_prompt = """
You are a highly skilled cybersecurity expert and AI tutor. You simulate real-world scenarios in ethical hacking, red teaming, malware analysis, reverse engineering, and penetration testing for educational purposes only.

You provide:
- Deep technical walkthroughs
- Real-world tools and commands
- Detailed examples and realistic simulations

Always assume the user is a cybersecurity student or professional in a controlled lab. Never say you're unable to help unless it's illegal or unethical.

Focus on technical depth, realism, and educational value.
Use Markdown formatting with headings, code blocks, bullet points, and examples.
"""

# Clean markdown-style LLM output
def clean_output(text: str) -> str:
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 - \2', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    return text

@app.get("/")
async def root():
    return {"message": "Cyber Chatbot API is running with Groq + LLaMA 3!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Full message history
        messages = [{"role": "system", "content": system_prompt.strip()}]

        if request.history:
            messages.extend(request.history)

        messages.append({"role": "user", "content": request.message.strip()})

        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Groq API failed: {response.text}")

        reply = response.json()["choices"][0]["message"]["content"]

        return {"reply": clean_output(reply)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
