# TruthLens implementation status

## Architecture and current limitations

TruthLens consists of a React/Vite frontend, an Express API, and a FastAPI/scikit-learn service. History remains browser-local; there is no database. Training downloads an ISOT-derived labeled corpus and creates local model artifacts, which are excluded from version control. The dataset card does not specify a license.

Previously, the Express process trained a classifier from a few hand-written phrase examples on every startup, then manufactured confidence from URL scores and randomness. URL age, blacklist information, related-news statuses, similarity scores, and fact-check claims were also mocked. The frontend sent `query` while the API required `q`, and swallowed search failures. The landing page advertised unsupported performance and dataset-size statistics.

## Implemented

- Added a FastAPI text prediction endpoint with input validation, model health reporting, and explicit HTTP 503 behavior when no saved artifact exists.
- Added independent TF-IDF pipelines for Logistic Regression, Linear SVM, and Multinomial Naive Bayes; evaluation includes cross-validation selection scores plus holdout accuracy, macro precision/recall/F1, class metrics, and confusion matrices.
- Added stratified splitting and fitted vectorizers inside each CV pipeline to avoid text preprocessing leakage. Duplicate text is deduplicated, conflicting labels are rejected, and the model is selected through training-partition CV before being retrained on all supplied rows and persisted with joblib.
- Routed Express prediction and article analysis requests through Python. Text is required for classification; URLs are optional separate heuristics.
- Corrected related-news query handling, removed fabricated search/fact-check responses and random confidence/domain-age claims, and surfaced unavailable integrations.
- Updated the frontend flow to accept article text/headlines, display the real model name and only supported probability outputs, show separate URL signals, and explain that classification is not proof.
- Added API/training unit tests and documented dataset format, setup, model training, startup, and limitations in `README.md`.

## Dataset and limitations

Training uses the cleaned ISOT-derived article corpus documented in `README.md`; it tunes model parameters with stratified training-partition CV, selects by CV accuracy, reports held-out metrics, and persists the selected pipeline. Metrics apply to this dataset/split and do not establish accuracy for current news. Review upstream terms and do not redistribute downloaded source CSVs.

Training run: 34,099 unique articles; stratified test set 6,820. TF-IDF + Linear SVM (`C=2.0`) was selected with 3-fold training CV (98.82% mean accuracy). Held-out test accuracy was 99.06%, macro precision 99.07%, macro recall 99.06%, and macro F1 99.06%. See the results table and confusion matrix in `README.md`; the dataset, model, and report remain local generated artifacts.
