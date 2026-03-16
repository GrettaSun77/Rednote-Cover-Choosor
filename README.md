# Xiaohongshu Cover MVP

This project builds a practical MVP for picking the best Xiaohongshu cover image from a set of candidates.

It combines three signals instead of trusting a single model:

1. OpenAI vision scoring for actual image understanding
2. Local image-feature scoring for brightness, contrast, sharpness, and thumbnail balance
3. Historical preference reranking from your Excel vote data

## Files

- `input.xlsx`: your historical vote summary workbook
- `build_dataset.py`: converts the Excel file into a normalized dataset
- `data/processed/training_dataset.json`: parsed training data
- `cover_selector.py`: multi-model scoring and score fusion
- `app.py`: Streamlit MVP UI

## How the MVP works

1. Parse the Excel workbook into a structured dataset
2. Extract the winner patterns from historical vote summaries
3. Upload a new set of candidate images
4. Run OpenAI vision scoring on the candidates
5. Blend that result with local image features and historical preference weights
6. Return the best cover recommendation with score breakdowns

## Setup

Create the dataset:

```powershell
python .\build_dataset.py
```

Install dependencies:

```powershell
pip install -r .\requirements.txt
```

Set your API key:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Optional:

```powershell
$env:OPENAI_VISION_MODEL="gpt-4.1-mini"
```

For Streamlit Community Cloud, add `OPENAI_API_KEY` in the app Secrets panel instead of hardcoding it.

Run the app:

```powershell
streamlit run .\app.py
```

## Current MVP scope

This version is intentionally simple but practical:

- It does use an external image model, not just the local spreadsheet
- It does use your historical winner data as a preference layer
- It does not yet train a custom ranking model
- It does not yet map each embedded Excel image back to its exact row automatically

## Best next upgrades

- Add an explicit batch-to-image mapping table
- Add a second external scorer such as CLIP, SigLIP, PickScore, or ImageReward
- Run backtests against your historical batches
- Add a comment-ingestion pipeline later
