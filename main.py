from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# --- Google Cloud credentials setup ---
key_path = "/etc/secrets/google-service-key.json"
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)

if not os.path.exists(key_path):
    raise FileNotFoundError(f"Google credentials file not found at {key_path}")

credentials = service_account.Credentials.from_service_account_file(key_path)
client = language_v1.LanguageServiceClient(credentials=credentials)

# --- Serve the UI ---
@app.get("/", response_class=HTMLResponse)
def serve_home():
    path = os.path.join("static", "app.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>Missing static/app.html</h1>", status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# --- NLP API endpoint ---
@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text.strip():
        return {"error": "No text provided"}

    doc = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    entities = client.analyze_entities(document=doc)
    sentiment = client.analyze_sentiment(document=doc)
    syntax = client.analyze_syntax(document=doc)

    results = {
        "entities": [
            {"name": e.name, "type": language_v1.Entity.Type(e.type_).name, "salience": round(e.salience, 3)}
            for e in entities.entities
        ],
        "sentiment": {
            "score": round(sentiment.document_sentiment.score, 3),
            "magnitude": round(sentiment.document_sentiment.magnitude, 3)
        },
        "syntax": [
            {"text": t.text.content, "pos": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name}
            for t in syntax.tokens[:50]
        ],
    }
    return results

# --- Fallback for any other route ---
@app.get("/{catch_all:path}", response_class=HTMLResponse)
def serve_html(catch_all: str):
    path = os.path.join("static", "app.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>app.html not found</h1>", status_code=404)
