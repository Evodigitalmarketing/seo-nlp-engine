from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from google.cloud import language_v1
from google.oauth2 import service_account
import os
import time

app = FastAPI()

# Debug route — confirms what Render actually sees
@app.get("/debug")
def debug_info():
    base = os.getcwd()
    tree = []
    for root, dirs, files in os.walk(".", topdown=True):
        for f in files:
            tree.append(os.path.join(root, f))
    info = f"WORKING DIR: {base}\n\nFILES:\n" + "\n".join(tree)
    return PlainTextResponse(info)

# Root route — will serve the app.html *after* we confirm the correct path
@app.get("/")
def root():
    return JSONResponse({"status": "ok", "message": "Diagnostic build active. Go to /debug"})

# --- Google Cloud NLP setup (keep working while we debug UI) ---
key_path = "/etc/secrets/google-service-key.json"
for _ in range(10):
    if os.path.exists(key_path):
        break
    time.sleep(1)
credentials = service_account.Credentials.from_service_account_file(key_path)
client = language_v1.LanguageServiceClient(credentials=credentials)

@app.post("/analyze")
async def analyze_text(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    result = client.analyze_sentiment(document=document)
    return {"sentiment": result.document_sentiment.score}



