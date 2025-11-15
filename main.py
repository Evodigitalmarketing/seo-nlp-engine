from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# --- Google Cloud credentials setup ---
key_path = "/etc/secrets/google-service-key.json"

# Wait for Render to mount the secret file
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)

if not os.path.exists(key_path):
    print(f"⚠️ Credentials file not found at {key_path}")
    credentials = None
else:
    print(f"✅ Credentials file found at {key_path}")
    credentials = service_account.Credentials.from_service_account_file(key_path)

client = language_v1.LanguageServiceClient(credentials=credentials) if credentials else None


# --- Serve homepage manually ---
@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the HTML interface."""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error loading page</h1><pre>{e}</pre>"


# --- NLP API endpoint ---
@app.post("/analyze")
async def analyze_text(request: Request):
    if not client:
        return {"error": "Google credentials not loaded on server."}

    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return {"error": "No text provided."}

    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Run NLP tasks
    entities = client.analyze_entities(document=document).entities
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    syntax = client.analyze_syntax(document=document).tokens

    categories = []
    if len(text.split()) > 20:
        try:
            category_response = client.classify_text(document=document)
            categories = [
                {"name": c.name, "confidence": round(c.confidence, 3)} for c in category_response.categories
            ]
        except Exception:
            categories = []

    # Structure output
    return {
        "entities": [
            {"name": e.name, "type": language_v1.Entity.Type(e.type_).name, "salience": round(e.salience, 3)}
            for e in entities
        ],
        "sentiment": {"score": round(sentiment.score, 3), "magnitude": round(sentiment.magnitude, 3)},
        "categories": categories,
        "syntax": [
            {"text": t.text.content, "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name}
            for t in syntax[:50]
        ],
    }



