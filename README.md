# SEO NLP Engine (Render + Google Cloud NLP)

## Overview
This FastAPI service connects to Google Cloud Natural Language API to extract entities, sentiment, categories, and syntax data from text.

## Deploying to Render
1. Create a service account in Google Cloud with access to 'Cloud Natural Language API'.
2. Download the JSON key and upload it to Render as a Secret File at:
   /etc/secrets/google-key.json
3. Set an environment variable:
   GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/google-key.json
4. Connect your GitHub repo to Render and deploy.

## Test locally
uvicorn main:app --reload

## Example request
POST https://your-app-name.onrender.com/analyze
Body:
{
  "text": "Google Cloud NLP helps developers understand text data."
}
