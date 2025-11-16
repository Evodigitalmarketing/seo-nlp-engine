from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import json

# ============================================================
#  FASTAPI APP SETUP
# ============================================================

app = FastAPI()

# ✅ Serve your frontend from the /static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Homepage (serves your UI)
@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("static/app.html", "r", encoding="utf-8") as f:
        return f.read()

# ✅ Render sends HEAD requests to check health — handle them here
@app.head("/")
async def head_home():
    # simple 200 OK response for Render’s health check
    return {"status": "ok"}

# ============================================================
#  GOOGLE CLOUD NLP CREDENTIALS
# ============================================================

creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not creds_json:
    raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")

# Parse the JSON key stored in Render environment variable
credentials_info = json.loads(creds_json)
credentials = service_account.Credentials.from_service_account_info(credentials_info)
client = language_v1.LanguageServiceClient(credentials=credentials)

# ============================================================
#  NLP ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text.strip():
        return {"error": "No text provided."}

    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )

    entities_response = client.analyze_entities(document=document)
    sentiment_response = client.analyze_sentiment(document=document)
    syntax_response = client.analyze_syntax(document=document)

    entities = [
        {
            "name": e.name,
            "type": language_v1.Entity.Type(e.type_).name,
            "salience": round(e.salience, 3)
        }
        for e in entities_response.entities
    ]

    sentiment = {
        "score": round(sentiment_response.document_sentiment.score, 3),
        "magnitude": round(sentiment_response.document_sentiment.magnitude, 3)
    }

    tokens = [
        {
            "text": t.text.content,
            "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name
        }
        for t in syntax_response.tokens
    ]

    return {
        "entities": entities,
        "sentiment": sentiment,
        "syntax": tokens[:50]
    }

# ============================================================
#  LOCAL DEV ENTRYPOINT (ignored on Render)
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
