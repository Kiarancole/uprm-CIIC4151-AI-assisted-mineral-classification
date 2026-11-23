# Web Component

This directory contains the Streamlit interface for uploading mineral photographs and displaying classification results.

## Current scope

The frontend can:

- accept and preview JPG or PNG images;
- send the uploaded image to the prediction API as base64 text;
- display the predicted mineral, confidence, and up to three candidates;
- explain supported minerals and important limitations; and
- report clear errors when the prediction service is unavailable.

The frontend does not load the PyTorch model directly. It expects a prediction service at `http://127.0.0.1:8000` by default and sends this request:

```json
{
  "uploaded_image": "<base64 image data>"
}
```

The preferred API response is:

```json
{
  "predicted_mineral": "quartz",
  "confidence": 0.87,
  "top_predictions": [
    {"mineral": "quartz", "confidence": 0.87},
    {"mineral": "calcite", "confidence": 0.08},
    {"mineral": "gypsum", "confidence": 0.03}
  ]
}
```

## Run locally

From the repository root:

```bash
python -m pip install -r src/web/requirements.txt
python -m streamlit run src/web/app.py
```

To use a prediction service at another address, set `MINERAL_API_URL` before starting Streamlit.

PowerShell:

```powershell
$env:MINERAL_API_URL = "https://example-api.invalid"
python -m streamlit run src/web/app.py
```

The page remains available for visual review when the API is offline, but prediction requests will display a connection error until the service is running.
