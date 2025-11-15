from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# --- Load Google credentials from secret ---
key_path = "/etc/secrets/google-service-key.json"

# Wait for Render to mount secrets (in case of cold start delay)
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)

if not os.path.exists(key_path):
    raise FileNotFoundError(f"Google service account key not found at {key_path}")

# Load credentials explicitly
credentials = service_account.Credentials.from_service_account_file(key_path)
client = language_v1.LanguageServiceClient(credentials=credentials)


# --- Serve front-end page ---
@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serves the index.html front-end UI."""
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r") as f:
                html = f.read()
            return HTMLResponse(content=html, status_code=200)
        else:
            return HTMLResponse(
                content="""
                <h2>SEO NLP Engine</h2>
                <p>Frontend not found. API endpoint available at <code>/analyze</code>.</p>
                """,
                status_code=200
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading home page: {e}")


# --- Analyze text using Google Cloud NLP ---
@app.post("/analyze")
async def analyze_text(request: Request):
    """Analyzes text for entities, sentiment, syntax, and categories."""
    try:
        data = await request.json()
        text = data.get("text", "").strip()

        if not text:
            return JSONResponse({"error": "No text provided."}, status_code=400)

        document = language_v1.Document(
            content=text, type_=language_v1.Document.Type.PLAIN_TEXT
        )

        # Run analyses
        entities_response = client.analyze_entities(document=document)
        sentiment_response = client.analyze_sentiment(document=document)
        category_response = (
            client.classify_text(document=document) if len(text.split()) > 20 else None
        )
        syntax_response = client.analyze_syntax(document=document)

        # Format results
        entities = [
            {
                "name": e.name,
                "type": language_v1.Entity.Type(e.type_).name,
                "salience": round(e.salience, 3),
            }
            for e in entities_response.entities
        ]

        sentiment = {
            "score": round(sentiment_response.document_sentiment.score, 3),
            "magnitude": round(sentiment_response.document_sentiment.magnitude, 3),
        }

        categories = (
            [
                {"name": c.name, "confidence": round(c.confidence, 3)}
                for c in category_response.categories
            ]
            if category_response
            else []
        )

        tokens = [
            {
                "text": t.text.content,
                "part_of_speech": language_v1.PartOfSpeech.Tag(
                    t.part_of_speech.tag
                ).name,
            }
            for t in syntax_response.tokens
        ]

        return JSONResponse(
            content={
                "entities": entities,
                "sentiment": sentiment,
                "categories": categories,
                "syntax": tokens[:50],
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
