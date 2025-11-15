from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# --- Always serve the HTML file from an absolute path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FILE = os.path.join(BASE_DIR, "static", "app.html")

@app.get("/", response_class=FileResponse)
def root():
    """Serve frontend HTML file or return JSON error if missing."""
    if os.path.exists(FRONTEND_FILE):
        return FileResponse(FRONTEND_FILE)
    return JSONResponse({"error": f"Frontend file not found at {FRONTEND_FILE}"}, status_code=404)


# --- Google Cloud credentials ---
key_path = "/etc/secrets/google-service-key.json"

# Wait for Render to mount the secret
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)

if not os.path.exists(key_path):
    raise FileNotFoundError(f"Google credentials not found at {key_path}")

# Load credentials and initialize NLP client
credentials = service_account.Credentials.from_service_account_file(key_path)
client = language_v1.LanguageServiceClient(credentials=credentials)


@app.get("/healthz")
def health_check():
    """Render health check endpoint"""
    return {"status": "ok", "message": "Google Cloud NLP API is connected."}


@app.post("/analyze")
async def analyze_text(request: Request):
    """Analyze text using Google Cloud NLP"""
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)

    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Run NLP analyses
    entities_response = client.analyze_entities(document=document)
    sentiment_response = client.analyze_sentiment(document=document)
    syntax_response = client.analyze_syntax(document=document)

    try:
        category_response = client.classify_text(document=document) if len(text.split()) > 20 else None
    except Exception:
        category_response = None

    # Format results
    entities = [
        {
            "name": e.name,
            "type": language_v1.Entity.Type(e.type_).name,
            "salience": round(e.salience, 3)
        } for e in entities_response.entities
    ]

    sentiment = {
        "score": round(sentiment_response.document_sentiment.score, 3),
        "magnitude": round(sentiment_response.document_sentiment.magnitude, 3),
    }

    categories = [
        {"name": c.name, "confidence": round(c.confidence, 3)}
        for c in category_response.categories
    ] if category_response else []

    tokens = [
        {
            "text": t.text.content,
            "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name,
        } for t in syntax_response.tokens[:50]
    ]

    return {
        "entities": entities,
        "sentiment": sentiment,
        "categories": categories,
        "syntax": tokens,
    }


