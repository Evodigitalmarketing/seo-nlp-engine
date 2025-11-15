from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# --- Google Cloud credentials setup ---
key_path = "/etc/secrets/google-service-key.json"

# Wait up to 10 seconds for Render to mount secrets
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)

if not os.path.exists(key_path):
    raise FileNotFoundError(f"Google credentials file not found at {key_path}")

# Load credentials and initialize client
credentials = service_account.Credentials.from_service_account_file(key_path)
client = language_v1.LanguageServiceClient(credentials=credentials)

# --- Serve the main UI (renamed app.html) ---
@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serve the UI file manually to avoid Render caching."""
    try:
        with open("app.html", "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading app.html</h1><pre>{e}</pre>", status_code=500)


# --- NLP Analysis Endpoint ---
@app.post("/analyze")
async def analyze_text(request: Request):
    """Analyze input text using Google Cloud NLP API."""
    data = await request.json()
    text = data.get("text", "")

    if not text.strip():
        return {"error": "No text provided"}

    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Run all analyses
    entities_response = client.analyze_entities(document=document)
    sentiment_response = client.analyze_sentiment(document=document)
    syntax_response = client.analyze_syntax(document=document)

    # Classify only if text is long enough
    category_response = None
    if len(text.split()) > 20:
        try:
            category_response = client.classify_text(document=document)
        except Exception:
            category_response = None

    # Structure response
    entities = [
        {"name": e.name, "type": language_v1.Entity.Type(e.type_).name, "salience": round(e.salience, 3)}
        for e in entities_response.entities
    ]

    sentiment = {
        "score": round(sentiment_response.document_sentiment.score, 3),
        "magnitude": round(sentiment_response.document_sentiment.magnitude, 3)
    }

    categories = [
        {"name": c.name, "confidence": round(c.confidence, 3)}
        for c in category_response.categories
    ] if category_response else []

    tokens = [
        {"text": t.text.content, "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name}
        for t in syntax_response.tokens
    ]

    return {
        "entities": entities,
        "sentiment": sentiment,
        "categories": categories,
        "syntax": tokens[:50]
    }


# --- Catch-all route to serve HTML for all paths ---
@app.get("/{catch_all:path}", response_class=HTMLResponse)
def serve_html(catch_all: str):
    """Serve the same app.html for any route."""
    if os.path.exists("app.html"):
        with open("app.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse("<h1>app.html not found</h1>", status_code=404)
