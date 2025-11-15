from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# === Serve the front-end ===
@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serve the HTML UI or fallback message if index.html missing."""
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r") as f:
                return f.read()
        else:
            return HTMLResponse(
                content="<h2>Welcome to the SEO NLP Engine!</h2><p>No index.html found — try the API at <code>/analyze</code>.</p>",
                status_code=200
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading home page: {e}")

# === NLP Analysis Endpoint ===
@app.post("/analyze")
async def analyze_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        if not text:
            return JSONResponse(content={"error": "No text provided."}, status_code=400)

        key_path = "/etc/secrets/google-service-key.json"

        # Wait briefly for Render to mount secrets
        for _ in range(10):
            if os.path.exists(key_path):
                break
            time.sleep(1)

        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Credentials file not found at {key_path}")

        # Authenticate and create client
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = language_v1.LanguageServiceClient(credentials=credentials)

        # Create the document for NLP
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

        # Run NLP analyses
        entities_response = client.analyze_entities(document=document)
        sentiment_response = client.analyze_sentiment(document=document)
        category_response = client.classify_text(document=document) if len(text.split()) > 20 else None
        syntax_response = client.analyze_syntax(document=document)

        # Structure the output
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
            for c in category_response.categories
        ] if category_response else []

        tokens = [
            {"text": t.text.content, "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name}
            for t in syntax_response.tokens[:50]
        ]

        return JSONResponse(content={
            "entities": entities,
            "sentiment": sentiment,
            "categories": categories,
            "syntax": tokens
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

