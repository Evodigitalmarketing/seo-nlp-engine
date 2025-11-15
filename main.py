from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()


# --- Google Cloud Credentials ---
KEY_PATH = "/etc/secrets/google-service-key.json"

# Wait for Render to mount the credentials file
for _ in range(10):
    if os.path.exists(KEY_PATH):
        break
    time.sleep(1)

if not os.path.exists(KEY_PATH):
    print(f"⚠️ Credentials file not found at {KEY_PATH}")
    credentials = None
else:
    print(f"✅ Loaded credentials from {KEY_PATH}")
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

client = language_v1.LanguageServiceClient(credentials=credentials) if credentials else None


# --- Serve index.html manually ---
@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the HTML frontend."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading index.html</h1><pre>{e}</pre>", status_code=500)


# --- NLP Endpoint ---
@app.post("/analyze")
async def analyze_text(request: Request):
    if not client:
        return {"error": "Google credentials not loaded on server."}

    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return {"error": "No text provided."}

    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    entities = client.analyze_entities(document=document).entities
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    syntax = client.analyze_syntax(document=document).tokens

    categories = []
    if len(text.split()) > 20:
        try:
            cat_response = client.classify_text(document=document)
            categories = [
                {"name": c.name, "confidence": round(c.confidence, 3)}
                for c in cat_response.categories
            ]
        except Exception:
            categories = []

    return {
        "entities": [
            {"name": e.name, "type": language_v1.Entity.Type(e.type_).name, "salience": round(e.salience, 3)}
            for e in entities
        ],
        "sentiment": {
            "score": round(sentiment.score, 3),
            "magnitude": round(sentiment.magnitude, 3)
        },
        "categories": categories,
        "syntax": [
            {"text": t.text.content, "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name}
            for t in syntax[:50]
        ],
    }


