# TruthLens — AI-Powered Fake News Detector

🚀 [**TRY LIVE DEMO — Open TruthLens**](https://fake-news-detector-kda6.onrender.com/)

---

## About the Project

TruthLens is a machine learning-based fake news detection web application designed to classify news articles as REAL or FAKE using Natural Language Processing (NLP) and supervised machine learning.

The platform combines a modern React interface, an Express backend, and a Python-based machine learning service powered by FastAPI and scikit-learn. It provides text classification, prediction results, optional URL analysis, prediction history, and integration with external news and fact-checking providers.

The application uses TF-IDF text vectorization and evaluates multiple machine learning algorithms to identify a suitable classifier for news text.

## Key Features

- **AI-Powered News Detection:** Classifies news text as REAL or FAKE using a trained ML model.
- **Multiple ML Algorithms:** Supports Logistic Regression, Linear SVM, and Multinomial Naive Bayes.
- **Natural Language Processing:** Converts news text into numerical features using TF-IDF.
- **Prediction Dashboard:** Displays classification results through an interactive web interface.
- **Prediction History:** Stores prediction history locally in the browser using localStorage.
- **URL Analysis:** Provides basic URL-related heuristic indicators.
- **News Search Integration:** Supports related-news searches through an external provider when configured.
- **Fact-Check Integration:** Supports external fact-checking queries when the required API key is configured.
- **Model Evaluation:** Evaluates classifiers using accuracy, precision, recall, macro-F1, and confusion matrices.
- **REST API Integration:** Connects the React frontend to the Express backend and Python ML service.
- **Responsive Interface:** Provides an accessible interface for entering news articles and viewing predictions.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React 18 |
| Frontend Tooling | Vite |
| Backend | Node.js, Express.js |
| ML Service | Python, FastAPI |
| Machine Learning | scikit-learn |
| NLP | TF-IDF Vectorization |
| Classification Algorithms | Logistic Regression, Linear SVM, Multinomial Naive Bayes |
| Model Serialization | Joblib |
| Prediction History | Browser localStorage |
| Deployment | Render |

## System Architecture

TruthLens follows a modular architecture in which the frontend, backend API, and machine learning service perform separate responsibilities.

```text
              User
               |
               v
       React + Vite Frontend
               |
               v
        Express.js Backend
               |
               v
        Python FastAPI API
               |
               v
        NLP Preprocessing
               |
               v
        TF-IDF Vectorizer
               |
               v
       Trained ML Classifier
               |
               v
       REAL / FAKE Prediction
               |
               v
       Display Result in UI
```

### Main Components

- **Frontend (`frontend/`):** Provides the user interface, news input, prediction display, and client-side interactions.
- **Express Backend (`backend/server.js`):** Handles API requests, validates input, provides URL-related heuristics, and forwards predictions to the Python service.
- **Python ML Service (`backend/ml_service/`):** Handles model inference, text classification, and model-related analysis.
- **Training Pipeline:** Prepares the dataset, trains and evaluates classifiers, and saves the selected model pipeline.

## Machine Learning Algorithms

TruthLens supports three supervised machine learning algorithms for binary news classification.

### 1. Logistic Regression

Uses TF-IDF text features to estimate the relationship between words and the REAL or FAKE labels. It provides an efficient baseline for text classification.

### 2. Linear Support Vector Machine

Uses a linear decision boundary to distinguish between REAL and FAKE news. It is a strong candidate for high-dimensional text classification tasks.

### 3. Multinomial Naive Bayes

Uses word-feature statistics and Bayes' theorem to estimate the class of a news article. It is computationally efficient and useful as a baseline model.

### Model Selection

The training pipeline compares the algorithms using cross-validation on the training partition. The selected algorithm is evaluated on a separate held-out test set before the final pipeline is saved.

## Dataset

The project uses a curated variant of the ISOT fake and real news dataset.

**Dataset source:** [ISOT Fake News Dataset FineTuned 2022](https://huggingface.co/datasets/Phoenyx83/ISOT-Fake-News-Dataset-FineTuned-2022)

The dataset contains news article text and corresponding labels for REAL and FAKE news.

The dataset preparation pipeline processes the source data, removes exact duplicate text, checks for conflicting labels, and prepares the cleaned dataset for model training.

## Model Training and Evaluation

The documented training run used the following configuration:

- **Unique usable articles:** 34,099 after exact-text deduplication.
- **Held-out test samples:** 6,820.
- **Selected algorithm:** TF-IDF + Linear SVM.
- **Cross-validation:** Three-fold evaluation on the training partition.

### Reported Model Performance

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 98.27% | 98.28% | 98.27% | 98.27% |
| Linear SVM | 99.06% | 99.07% | 99.06% | 99.06% |
| Multinomial Naive Bayes | 95.16% | 95.17% | 95.16% | 95.16% |

The reported results come from the documented training run dated October 26, 2026. They should be reproducible from the same dataset and training configuration.

## Installation and Setup

### Prerequisites

- Python 3.10 or newer.
- Node.js 18 or newer.
- npm.
- Git.

### 1. Clone the Project

Clone the repository and open the project directory in your terminal.

### 2. Set Up the Python Environment

From the project root, run:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Prepare the Dataset

```powershell
python -m ml_service.prepare_isot
```

### 4. Train the Models

```powershell
python -m ml_service.train --data .\ml_service\data\isot_fake_news.csv
```

The training pipeline compares the supported algorithms, evaluates their performance, and saves the selected model and preprocessing pipeline.

### 5. Start the Python ML Service

```powershell
python -m uvicorn ml_service.main:app --host 127.0.0.1 --port 8001
```

Keep this terminal running.

### 6. Start the Express Backend

Open a second terminal:

```powershell
cd backend
npm install
$env:PORT='5001'
$env:ML_SERVICE_URL='http://127.0.0.1:8001'
npm run dev
```

### 7. Start the Frontend

Open a third terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, typically `http://localhost:5173`.

Ensure the frontend API proxy is configured to use the correct Express backend port.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Classifies news text as REAL or FAKE. |
| POST | `/api/analyze` | Analyzes news text with optional URL indicators. |
| GET | `/api/news?q=...` | Searches related news when configured. |
| GET | `/api/factcheck?q=...` | Queries an external fact-checking provider when configured. |

### Example Prediction Request

```json
{
  "text": "The news article text to classify"
}
```

The prediction endpoint returns the predicted label and model information. Probability estimates are returned when supported by the selected model.

## Testing

Run the backend tests:

```powershell
cd backend
npm test
```

Run the Python ML service tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ml_service\tests
```

Build the frontend:

```powershell
cd frontend
npm run build
```

## Future Enhancements

- Transformer-based NLP models for comparative evaluation.
- Improved generalization to recent and previously unseen news.
- Enhanced explainability for individual predictions.
- Evaluation across multiple independent news datasets.
- Additional automated testing and production monitoring.
- Improved integration with reliable external fact-checking sources.
