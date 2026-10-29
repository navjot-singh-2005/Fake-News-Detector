import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "artifacts" / "classifier.joblib"))
EVALUATION_REPORT_PATH = BASE_DIR / "artifacts" / "evaluation_report.json"
MAX_TEXT_LENGTH = 100_000

app = FastAPI(title="TruthLens ML API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5001,http://127.0.0.1:5001,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model_artifact: dict[str, Any] | None = None


def load_model() -> None:
    global model_artifact
    if MODEL_PATH.is_file():
        artifact = joblib.load(MODEL_PATH)
        if not isinstance(artifact, dict) or "pipeline" not in artifact:
            raise RuntimeError(f"Invalid model artifact at {MODEL_PATH}")
        model_artifact = artifact
    else:
        model_artifact = None


@app.on_event("startup")
def startup() -> None:
    load_model()


def compute_probabilities(pipeline: Any, text: str) -> dict[str, float] | None:
    """Calculate calibrated class probabilities for any pipeline.

    Supports native predict_proba, decision_function with sigmoid calibration (e.g. LinearSVC),
    and calibrated classifiers.
    """
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        classes = getattr(pipeline, "classes_", ["FAKE", "REAL"])
        return {
            str(class_name): round(float(prob), 4)
            for class_name, prob in zip(classes, probabilities)
        }

    if hasattr(pipeline, "decision_function"):
        try:
            decision = float(pipeline.decision_function([text])[0])
            classes = list(getattr(pipeline, "classes_", ["FAKE", "REAL"]))
            # Sigmoid probability for binary classification
            prob_positive = 1.0 / (1.0 + math.exp(-decision))
            if len(classes) == 2:
                # If classes_[1] is 'REAL', prob_positive is probability of REAL
                if classes[1] == "REAL":
                    return {
                        "FAKE": round(1.0 - prob_positive, 4),
                        "REAL": round(prob_positive, 4),
                    }
                else:
                    return {
                        str(classes[0]): round(1.0 - prob_positive, 4),
                        str(classes[1]): round(prob_positive, 4),
                    }
        except Exception:
            return None

    return None


def extract_top_signals(pipeline: Any, text: str, top_k: int = 5) -> dict[str, list[str]]:
    """Extract top indicative words and bigrams influencing the classification."""
    try:
        tfidf = pipeline.named_steps.get("tfidf")
        classifier = pipeline.named_steps.get("classifier")
        if not tfidf or not classifier:
            return {"fake": [], "real": []}

        feature_names = np.array(tfidf.get_feature_names_out())
        x_vec = tfidf.transform([text])
        coo = x_vec.tocoo()
        if coo.nnz == 0:
            return {"fake": [], "real": []}

        coefs = None
        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]
        elif hasattr(classifier, "feature_log_prob_"):
            # For Naive Bayes, log prob difference between class 1 (REAL) and class 0 (FAKE)
            coefs = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

        if coefs is None:
            return {"fake": [], "real": []}

        contributions: list[tuple[str, float]] = []
        for col_idx, val in zip(coo.col, coo.data):
            feat = str(feature_names[col_idx])
            weight = float(coefs[col_idx] * val)
            contributions.append((feat, weight))

        # Negative weights lean FAKE; positive weights lean REAL
        contributions.sort(key=lambda item: item[1])
        fake_words = [feat for feat, w in contributions if w < -0.02][:top_k]
        real_words = [feat for feat, w in reversed(contributions) if w > 0.02][:top_k]
        return {"fake": fake_words, "real": real_words}
    except Exception:
        return {"fake": [], "real": []}


def analyze_text_metrics(text: str) -> dict[str, Any]:
    """Compute linguistic metrics and sensationalism indicators."""
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return {
            "word_count": 0,
            "sensationalism_score": 0,
            "uppercase_ratio": 0.0,
            "exclamation_count": 0,
            "detected_cues": [],
        }

    uppercase_words = sum(1 for w in words if len(w) > 1 and w.isupper())
    uppercase_ratio = round(uppercase_words / total_words, 3)
    exclamation_count = text.count("!")

    clickbait_cues = [
        "shocking",
        "urgent",
        "secret",
        "exposed",
        "breaking",
        "miracle",
        "won't believe",
        "conspiracy",
        "unbelievable",
        "hoax",
        "alert",
    ]
    text_lower = text.lower()
    cues_found = [cue for cue in clickbait_cues if cue in text_lower]

    sensationalism = min(
        100,
        int(
            (uppercase_ratio * 45)
            + (min(exclamation_count, 5) * 8)
            + (len(cues_found) * 12)
        ),
    )

    return {
        "word_count": total_words,
        "sensationalism_score": sensationalism,
        "uppercase_ratio": uppercase_ratio,
        "exclamation_count": exclamation_count,
        "detected_cues": cues_found,
    }


def extract_url_features(raw_url: str | None) -> dict[str, Any] | None:
    """Analyze domain reputation and URL heuristics."""
    if not raw_url:
        return None
    try:
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None

        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        suspicious_tlds = [
            ".xyz",
            ".click",
            ".info",
            ".tk",
            ".ml",
            ".ga",
            ".cf",
            ".gq",
            ".top",
            ".buzz",
            ".work",
        ]
        suspicious_keywords = [
            "clickbait",
            "shocking",
            "viral",
            "exposed",
            "truth",
            "secret",
            "miracle",
            "alert",
            "breaking",
            "urgent",
            "weird",
        ]
        recognized_news_domains = [
            "bbc.com",
            "reuters.com",
            "apnews.com",
            "theguardian.com",
            "nytimes.com",
            "washingtonpost.com",
            "bloomberg.com",
            "npr.org",
            "thehindu.com",
            "ndtv.com",
            "hindustantimes.com",
            "timesofindia.com",
            "aljazeera.com",
            "cnn.com",
            "wsj.com",
        ]

        is_https = parsed.scheme == "https"
        is_trusted = any(
            domain == known or domain.endswith(f".{known}")
            for known in recognized_news_domains
        )
        has_suspicious_tld = any(domain.endswith(tld) for tld in suspicious_tlds)
        found_keywords = [kw for kw in suspicious_keywords if kw in raw_url.lower()]
        path_depth = len([p for p in parsed.path.split("/") if p])
        flags: list[str] = []

        if not is_https:
            flags.append("The URL does not use HTTPS encryption.")
        if has_suspicious_tld:
            flags.append("The URL uses a commonly abused top-level domain.")
        if found_keywords:
            flags.append(f"Potentially sensational URL terms: {', '.join(found_keywords)}.")
        if len(domain) > 30:
            flags.append("The domain name is unusually long.")

        trust_score = 50 + (15 if is_https else -15) + (20 if is_trusted else 0)
        if has_suspicious_tld:
            trust_score -= 15
        trust_score -= len(found_keywords) * 5
        if len(domain) > 30:
            trust_score -= 5
        trust_score = max(0, min(100, round(trust_score)))

        return {
            "domain": domain,
            "isHttps": is_https,
            "isTrusted": is_trusted,
            "domainAge": None,
            "isBlacklisted": None,
            "urlLength": len(raw_url),
            "pathDepth": path_depth,
            "hasSuspiciousTLD": has_suspicious_tld,
            "suspiciousKeywords": found_keywords,
            "trustScore": trust_score,
            "flags": flags,
        }
    except Exception:
        return None


def make_explanation(label: str, model_name: str, has_url: bool, signals: dict[str, list[str]]) -> str:
    verdict = "classified as REAL" if label == "REAL" else "classified as FAKE"
    explanation = f"The {model_name} model {verdict} based on text TF-IDF vector analysis."

    key_terms = signals.get("real" if label == "REAL" else "fake", [])
    if key_terms:
        explanation += f" Significant indicative terms identified: {', '.join(key_terms)}."

    if has_url:
        explanation += " URL indicators are separate heuristics and do not establish whether the article is true."

    explanation += " This machine-learning result is not definitive proof; always verify critical claims with multiple independent sources."
    return explanation


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Article text must contain non-whitespace characters.")
        return value


def scrape_article_from_url(url: str) -> tuple[str, str]:
    """Scrape article title and text from URL, with fallback to URL slug tokenization."""
    import re
    from urllib.parse import urlparse
    import urllib.request
    from bs4 import BeautifulSoup

    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[-1].replace("-", " ").replace("_", " ").replace(".html", "").replace(".htm", "")
    slug_text = re.sub(r"\d+", "", slug).strip()
    fallback_title = slug_text.capitalize() if slug_text else (parsed.netloc or "News Article")
    fallback_text = f"{fallback_title}. News report published on {parsed.netloc or url}."

    if not url.startswith(("http://", "https://")):
        return fallback_title, fallback_text

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")

            for elem in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                elem.extract()

            title = ""
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
            else:
                title = fallback_title

            description = ""
            og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
            if og_desc and og_desc.get("content"):
                description = og_desc["content"].strip()

            paragraphs = [
                p.get_text().strip()
                for p in soup.find_all("p")
                if len(p.get_text().strip()) > 35
            ]
            body_text = " ".join(paragraphs[:8])

            full_text = f"{title}. {description}. {body_text}".strip()
            if len(full_text) > 40:
                return title, full_text[:MAX_TEXT_LENGTH]
    except Exception:
        pass

    return fallback_title, fallback_text


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str | None = None
    url: str | None = None

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str | None:
        if isinstance(value, str):
            val = value.strip()
            if not val:
                return None
            if len(val) > MAX_TEXT_LENGTH:
                raise ValueError(f"Article text must be no longer than {MAX_TEXT_LENGTH} characters.")
            return val
        return None

    @field_validator("url", mode="before")
    @classmethod
    def clean_url(cls, value: Any) -> str | None:
        if isinstance(value, str):
            val = value.strip()
            if not val:
                return None
            parsed = urlparse(val)
            if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
                raise ValueError("Article URL must use the HTTP or HTTPS protocol.")
            return val
        return None


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model_artifact is not None,
        "model_name": model_artifact.get("model_name") if model_artifact else None,
        "algorithms_available": [
            "TF-IDF + Linear SVM (Calibrated)",
            "TF-IDF + Logistic Regression",
            "TF-IDF + Multinomial Naive Bayes",
            "TF-IDF + Passive-Aggressive Classifier",
            "TF-IDF + Soft-Voting Ensemble",
        ],
    }


@app.get("/api/models")
def get_models() -> dict[str, Any]:
    """Return model benchmarks, training parameters, and performance report."""
    report: dict[str, Any] = {}
    if EVALUATION_REPORT_PATH.is_file():
        try:
            import json
            report = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "active_model": model_artifact.get("model_name") if model_artifact else None,
        "model_loaded": model_artifact is not None,
        "dataset_rows": report.get("training_rows", 34099),
        "evaluation_report": report,
    }


ARTIFACT_PATTERNS = [
    re.compile(r"^[A-Z\s,]+?\s*\((Reuters|AP|AFP)\)\s*[-—]\s*", re.IGNORECASE),
    re.compile(r"\b(Reuters|Associated Press)\b", re.IGNORECASE),
    re.compile(r"\b(featured image|image via|read more|via)\b", re.IGNORECASE),
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
]


def clean_publisher_artifacts(text: str) -> str:
    """Clean publisher wire bylines and web scrap artifacts during inference to match training."""
    cleaned = text
    for pattern in ARTIFACT_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def classify_text(
    pipeline: Any,
    text: str,
    metrics: dict[str, Any] | None = None,
    url_features: dict[str, Any] | None = None,
) -> tuple[str, dict[str, float] | None, dict[str, list[str]]]:
    """Classify text using identical preprocessing, calibrated probabilities, and unbiased thresholding."""
    cleaned = clean_publisher_artifacts(text)
    eval_text = cleaned if len(cleaned) >= 5 else text
    probabilities = compute_probabilities(pipeline, eval_text)

    if metrics is None:
        metrics = analyze_text_metrics(text)

    prob_fake = probabilities.get("FAKE", 0.5) if probabilities else 0.5
    cues = len(metrics.get("detected_cues", []))
    sens = metrics.get("sensationalism_score", 0)
    is_trusted_domain = url_features.get("isTrusted", False) if url_features else False

    # Bias-Mitigated Calibrated Decision Threshold:
    # 1. Obvious/confident fake news or text with clickbait triggers: FAKE
    # 2. Neutral modern news from trusted domains or with low sensationalism (<10%) and zero clickbait:
    #    Require strong model confidence (prob_fake >= 0.82) to avoid false-positive FAKE on non-US news.
    if is_trusted_domain:
        fake_threshold = 0.85
    elif cues >= 1 or sens >= 20:
        fake_threshold = 0.52
    elif sens <= 8 and cues == 0:
        fake_threshold = 0.80
    else:
        fake_threshold = 0.65

    label = "FAKE" if prob_fake >= fake_threshold else "REAL"
    signals = extract_top_signals(pipeline, eval_text)
    return label, probabilities, signals


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is available. Train a model with a labeled news dataset first.",
        )

    pipeline = model_artifact["pipeline"]
    label, probabilities, signals = classify_text(pipeline, request.text)
    metrics = analyze_text_metrics(request.text)

    result: dict[str, Any] = {
        "label": label,
        "model_name": model_artifact["model_name"],
    }
    if probabilities:
        result["probabilities"] = probabilities

    result["top_signals"] = signals
    result["text_metrics"] = metrics
    return result


@app.post("/api/analyze")
def analyze(request: AnalysisRequest) -> dict[str, Any]:
    """Unified full analysis endpoint supporting both text and auto-scraped URLs."""
    if model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is available. Train a model with a labeled news dataset first.",
        )

    if not request.text and not request.url:
        raise HTTPException(
            status_code=400,
            detail="Please provide news text or a valid article URL to classify.",
        )

    extracted_title = None
    extracted_from_url = False
    text_to_analyze = request.text

    if not text_to_analyze and request.url:
        extracted_title, text_to_analyze = scrape_article_from_url(request.url)
        extracted_from_url = True

    if not text_to_analyze or not text_to_analyze.strip():
        text_to_analyze = f"Article from {request.url}"

    metrics = analyze_text_metrics(text_to_analyze)
    url_features = extract_url_features(request.url)
    pipeline = model_artifact["pipeline"]
    label, probabilities, signals = classify_text(
        pipeline, text_to_analyze, metrics=metrics, url_features=url_features
    )

    confidence = None
    if probabilities:
        confidence = round(max(probabilities.values()) * 100)

    credibility_breakdown = None
    if url_features or confidence is not None:
        credibility_breakdown = {
            "domainTrust": url_features.get("trustScore") if url_features else 50,
            "httpsScore": 100 if (url_features and url_features.get("isHttps")) else (0 if url_features else 50),
            "contentScore": confidence if confidence is not None else 50,
            "domainAge": None,
            "urlClean": max(0, 100 - len(url_features.get("suspiciousKeywords", [])) * 20) if url_features else 80,
        }

    explanation = make_explanation(
        label,
        model_artifact["model_name"],
        bool(url_features),
        signals,
    )

    return {
        "url": request.url or None,
        "urlFeatures": url_features,
        "extractedText": text_to_analyze if extracted_from_url else None,
        "extractedTitle": extracted_title,
        "extractedFromUrl": extracted_from_url,
        "mlResult": {
            "label": label,
            "modelName": model_artifact["model_name"],
            "confidence": confidence,
            "probabilities": probabilities,
            "topSignals": signals,
            "textMetrics": metrics,
        },
        "finalScore": confidence,
        "finalLabel": label,
        "explanation": explanation,
        "credibilityBreakdown": credibility_breakdown,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/news")
def related_news(q: str | None = None, query: str | None = None) -> dict[str, Any]:
    """Search related news articles or return clean fallback."""
    search_query = q or query
    if not search_query or not search_query.strip():
        raise HTTPException(status_code=400, detail="A search query is required.")

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return {
            "articles": [],
            "status": "Related-news search is unavailable; configure NEWS_API_KEY to enable live cross-verification.",
        }

    try:
        import urllib.request
        import json
        from urllib.parse import urlencode

        params = urlencode({
            "q": search_query.strip(),
            "language": "en",
            "pageSize": 5,
            "sortBy": "relevancy",
            "apiKey": api_key,
        })
        url = f"https://newsapi.org/v2/everything?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "TruthLens/2.0"})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = [
                {
                    "title": art.get("title"),
                    "source": art.get("source", {}).get("name", "Unknown"),
                    "url": art.get("url"),
                }
                for art in data.get("articles", [])
            ]
            return {"articles": articles}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Related news query failed: {exc}")
