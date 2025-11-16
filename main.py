from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# Serve all static files (like CSS/JS) if you add them later
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve app.html directly
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = "static/app.html"
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>Missing app.html file</h1>", status_code=500)
    with open(html_path, "r") as f:
        return f.read()


# Google Cloud NLP setup
key_path = "/etc/secrets/google-service-key.json"
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)

if not os.path.exists(key_path):
    raise FileNotFoundError(f"Credentials file not found: {key_path}")

credentials = service_account.Credentials.from_service_account_file(key_path)
client = language_v1.LanguageServiceClient(credentials=credentials)


@app.post("/analyze")
async def analyze_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)

        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

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
                "part_of_speech": language_v1.PartOfSpeech.Tag(t.part_of_speech.tag).name,
            }
            for t in syntax_response.tokens
        ]

        return {"entities": entities, "sentiment": sentiment, "syntax": tokens[:50]}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)




