from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# Serve static index.html from root
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Google Cloud credentials setup
key_path = "/etc/secrets/google-service-key.json"

# Wait for Render to mount the secret file
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)

if not os.path.exists(key_path):
    print(f"⚠️ Credentials file not found at {key_path}")
else:
    print(f"✅ Credentials file loaded: {key_path}")

# Load credentials once globally
credentials = None
if os.path.exists(key_path):
    credentials = service_account.Credentials.from_service_account_file(key_path)
client = language_v1.LanguageServiceClient(credentials=credentials) if credentials else None


@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serve the frontend."""
    with open("index.html", "r") as f:
        return f.read()


@app.post("/analyze")
async def analyze_text(request: Request):
    """Analyze text using Google Cloud NLP."""
    if not client:
        return {"error": "Google credentials not loaded on server."}

    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return {"error": "No text provided."}

    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    entities_response = client.analyze_entities(document=document)
    sentiment_response = client.analyze_sentiment(document=document)
    category_response = client.classify_text(document=document) if len(text.split()) > 20 else None
    syntax_response = client.analyze_syntax(document=document)

    entities = [
        {"name": e.name, "type": language_v1.Entity.Type(e.type_).name, "salience": round(e.salience, 3)}
        for e in entities_response.entities
    ]

    sentiment = {
        "score": round(sentiment_response.document_sentiment.score, 3),
        "magnitude": round(sentiment_response.document_sentiment.magnitude, 3),
    }

    categories = [
        {"name": c.name, "confidence": round(c.confidence, 3)}
        for c in (category_response.categories if category_response else [])
    ]

    syntax = [
        {"text": t.text.content, "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name}
        for t in syntax_response.tokens
    ]

    return {"entities": entities, "sentiment": sentiment, "categories": categories, "syntax": syntax[:50]}


