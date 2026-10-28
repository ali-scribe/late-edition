# Late Edition

Late Edition corrects OCR text from historic newspaper pages with a Gemma 4 model through Google's hosted Gemini API. It checks every proposed edit with a deterministic guard before accepting it.

The Gemma models are described as open-weight. Check the model card for the exact model you select to confirm its license and usage terms; this README does not independently verify the model-card license.

## What it does

The program takes a page image and its OCR text, splits the OCR text into chunks, sends the text to the selected model, checks the response, and writes an HTML report. The model receives OCR text only; it does not inspect or compare the page scan.

## Limitations

- Inference uses Google's hosted Gemini API. OCR text is sent to that service, so model inference is not offline or private.
- Numbers are never repaired by design. If a proposed edit changes any digit, the guard rejects that correction and keeps the original chunk.
- The included `sample_data/` files are sample data of unverified origin, not a verified real historic scan. Do not cite the sample as a historical source.
- Testing has used only a small sample. Guard thresholds have not been tuned or validated against a broad corpus.
- Free-tier API limits apply, and API calls can occasionally fail with 429 or 500-series errors.

## How the guard works

The guard compares each proposed correction with its original OCR chunk and rejects the whole chunk if any of these rules are violated:

1. **Digits must stay the same.** Adding, removing, changing, or reordering digits causes rejection.
2. **No new capitalized names or places.** A capitalized word not found in the original text may be flagged as an invented entity.
3. **The text must remain sufficiently similar.** A similarity score below 0.80 causes rejection.

When a chunk is rejected, the original OCR text is retained and the reason is included in the report.

## Setup

Create and activate a virtual environment, then install the project requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a local `.env` file and put your own Gemini API key and selected model ID there:

```env
GEMINI_API_KEY=your_api_key_here
GEMMA_MODEL=your_model_id_here
```

Keep the real key in `.env`, never in `.env.example` or a committed file. Choose a model ID and check its model card and availability for your account.

## How to run

Use the included sample files as local input:

```powershell
python cli.py run --offline sample_data/
```

This command still sends OCR text to the hosted Gemini API for inference. The `--mock` option substitutes a local mock correction for demonstrations; it does not check the selected model:

```powershell
python cli.py run --offline sample_data/ --mock
```

To save the report at a different location:

```powershell
python cli.py run --offline sample_data/ --output output/daily_report.html
```

## Use your own page

Download a pre-1931 page image and its OCR text from Chronicling America. Put one image file and one UTF-8 `.txt` OCR file in a folder. You may also add a `metadata.json` file using the same keys as `sample_data/metadata.json` (`date`, `title`, `lccn`, `sequence`, and `edition`). Run the program with that folder as the `--offline` input path. Despite the option name, correction still requires the hosted Gemini API unless you use `--mock`.

## Tests

Run the test suite with:

```powershell
.\.venv\Scripts\pytest -v
```
