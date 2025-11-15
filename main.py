from fastapi import FastAPI, Request
from google.cloud import language_v1
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "Google Cloud NLP API is connected."}

@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    text = data.get("text")

    if not text:
        return {"error": "No text provided."}

   from google.oauth2 import service_account

# Explicitly load credentials from the mounted Render secret file
credentials = service_account.Credentials.from_service_account_file(
    "/etc/secrets/google-service-key.json"
)
client = language_v1.LanguageServiceClient(credentials=credentials)

    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Run all analyses
    entities_response = client.analyze_entities(document=document)
    sentiment_response = client.analyze_sentiment(document=document)
    category_response = client.classify_text(document=document) if len(text.split()) > 20 else None
    syntax_response = client.analyze_syntax(document=document)

    # Structure output
    entities = [
        {
            "name": e.name,
            "type": language_v1.Entity.Type(e.type_).name,
            "salience": round(e.salience, 3)
        } for e in entities_response.entities
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
        {
            "text": t.text.content,
            "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name,
        }
        for t in syntax_response.tokens
    ]

    return {
        "entities": entities,
        "sentiment": sentiment,
        "categories": categories,
        "syntax": tokens[:50]
    }
from fastapi import FastAPI, Request
from google.cloud import language_v1
import os

# --- TEMP DIAGNOSTIC ---
print("DEBUG: Checking for /etc/secrets contents...")
if os.path.exists("/etc/secrets"):
    print("DEBUG: /etc/secrets exists. Files:", os.listdir("/etc/secrets"))
else:
    print("DEBUG: /etc/secrets does not exist.")
# --- END DIAGNOSTIC ---

