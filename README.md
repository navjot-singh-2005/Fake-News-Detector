# TruthLens

TruthLens is a React/Vite news-text classification UI backed by an Express API and a separate FastAPI/scikit-learn service. The Python service serves a saved TF-IDF pipeline; it does not train during API startup. A model is unavailable until one has been trained from a real labeled dataset.

The classifier predicts **REAL** or **FAKE** according to its training labels. It does not verify claims, establish truth, or replace independent fact-checking. Optional URL indicators are basic heuristics and are not evidence about an article.

## Architecture

- `frontend/`: React 18 and Vite application. Vite proxies `/api` requests to Express at `http://localhost:5000`.
- `backend/server.js`: Express API, input validation, URL heuristics, and forwarding to the Python prediction service.
- `backend/ml_service/`: FastAPI prediction API and scikit-learn training/evaluation code.
- Prediction history is stored in browser `localStorage`; article text is not stored in history. There is no database.

## Requirements

- Node.js 18 or newer and npm.
- Python 3.10 or newer.
- The ISOT-derived news dataset, prepared locally with `python -m ml_service.prepare_isot`.

## Dataset

The project uses a cleaned/curated variant of the commonly used ISOT fake/real news article corpus, hosted at [Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022](https://huggingface.co/datasets/Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022). The source CSVs contain article text and `type` labels (`0` real, `1` fake). Prepare these files locally; downloads are checked against their published content hashes:

```powershell
cd backend
python -m ml_service.prepare_isot
```

The upstream dataset card does not declare a specific usage license. Review the dataset and original ISOT terms before using it beyond local evaluation; do not redistribute source CSVs. Downloaded data and generated artifacts are excluded from version control.

## Install and train (Windows PowerShell)

From the project directory:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ml_service.prepare_isot
python -m ml_service.train --data .\ml_service\data\isot_fake_news.csv
```

Training removes exact duplicate text and rejects conflicting duplicate labels, then makes a stratified 80/20 train/test split. It tunes classifier regularization/smoothing with three-fold accuracy cross-validation on the training partition only, and evaluates each tuned pipeline on the held-out test partition. Each fold fits its own TF-IDF vectorizer, preventing preprocessing leakage. It independently evaluates:

1. TF-IDF + Logistic Regression.
2. TF-IDF + Linear SVM.
3. TF-IDF + Multinomial Naive Bayes.

It reports CV accuracy and held-out accuracy, macro precision, macro recall, macro F1, class-level metrics, and confusion matrices for each model. The algorithm with the highest mean training-partition CV accuracy is selected, then retrained on all cleaned labeled rows and saved to `backend/ml_service/artifacts/classifier.joblib`; test metrics are written to `evaluation_report.json`. Accuracy on this older curated dataset does not guarantee performance on current or out-of-domain news.

### Training run (2026-10-26)

The prepared dataset had **34,099 unique usable articles** after exact-text deduplication. Evaluation used a stratified held-out test set of **6,820 articles**. The selected pipeline was **TF-IDF + Linear SVM** with `C=2.0`, selected using training-only 3-fold CV (mean accuracy **98.82%**, standard deviation **0.04 percentage points**).

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 98.27% | 98.28% | 98.27% | 98.27% |
| TF-IDF + Linear SVM (selected) | **99.06%** | **99.07%** | **99.06%** | **99.06%** |
| TF-IDF + Multinomial Naive Bayes | 95.16% | 95.17% | 95.16% | 95.16% |

Selected-model confusion matrix (rows = actual, columns = predicted; order `FAKE`, `REAL`): `[[3362, 47], [17, 3394]]`. The saved evaluation report contains class-level metrics and other model details. Continuous calibrated probabilities are provided for all models (including Linear SVM via sigmoid decision-function scaling), alongside keyword feature explainability and linguistic metrics.

## Quick Start (One-Click)

Simply double-click `start.bat` in the root directory (or in `New folder/start.bat`). It starts the Python ML service and the frontend, then opens your browser at `http://localhost:5173`.

## Manual Setup & Run (Windows PowerShell)

Start the services in PowerShell:

**Terminal 1 — Python ML Service**

```powershell
cd backend
python -m uvicorn ml_service.main:app --host 127.0.0.1 --port 8001 --reload
```

**Terminal 2 — React Frontend**

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Python service handles both `/api/predict` and `/api/analyze` directly, with complete class probabilities, model feature contributions (indicative keywords), and linguistic metrics.

*(Optional)* If you wish to use the Express gateway on port 5001 as well:

```powershell
cd backend
$env:PORT='5001'
$env:ML_SERVICE_URL='http://127.0.0.1:8001'
npm install
npm run dev
```

## API

`POST /api/predict` (Express forwards the request to FastAPI):

```json
{
  "text": "The article text to classify"
}
```

Response fields include the predicted `label` and `model_name`. `probabilities` are returned only for models that expose them; they are model estimates, not calibrated truth or proof. Empty input and text longer than 100,000 characters are rejected. When no trained pipeline exists, the endpoint returns HTTP 503 with an explanatory message.

`POST /api/analyze` accepts the same required `text` plus an optional `url`; URL-derived heuristics remain separate from the text prediction. `GET /api/news?q=...` and `GET /api/factcheck?q=...` require their respective provider API keys; provider results are not labeled as confirmations by this app.

## Configuration

If `backend/.env` does not exist, copy `backend/.env.example` to create it; if it already exists, add or edit values without overwriting existing settings. `PORT`, `ML_SERVICE_URL`, `NEWS_API_KEY`, and `FACT_CHECK_API_KEY` are optional. The Vite proxy defaults to port 5001; if Express uses a different port, set `VITE_API_URL` in `frontend/.env` (see `frontend/.env.example`). `MODEL_PATH` and `CORS_ORIGINS` are optional Python process environment variables; by default the model is loaded from `backend/ml_service/artifacts/classifier.joblib` and browser CORS allows the local Vite origin.

## Tests and build

```powershell
cd backend
npm test
.\.venv\Scripts\python.exe -m unittest discover -s ml_service\tests
cd ..\frontend
npm run build
```

The backend/API tests include invalid, malformed, overlong, and unavailable-service cases. Training writes its actual evaluation report locally; the report, trained model, and dataset are excluded from version control and are not distributed with a fresh project copy.

## Limitations

- Dataset files, trained model, and evaluation report are downloaded/generated locally and are not committed.
- URL/domain indicators do not fetch or verify article content, domain age, or blacklist records.
- Related-news search and fact-check claims require configured external provider keys; related search results are not proof that a claim is confirmed or false.
- A model classification can be wrong, biased, or out of domain. Verify important information against reliable independent sources.
