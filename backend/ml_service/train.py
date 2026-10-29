import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABEL_ALIASES = {
    "fake": "FAKE",
    "false": "FAKE",
    "real": "REAL",
    "true": "REAL",
}
RANDOM_STATE = 42

ARTIFACT_PATTERNS = [
    re.compile(r"^[A-Z\s,]+?\s*\((Reuters|AP|AFP)\)\s*[-—]\s*", re.IGNORECASE),
    re.compile(r"\b(Reuters|Associated Press)\b", re.IGNORECASE),
    re.compile(r"\b(featured image|image via|read more|via)\b", re.IGNORECASE),
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
]


def clean_publisher_artifacts(text: str) -> str:
    """Strip publisher bylines and scraping artifacts that cause label shortcuts."""
    cleaned = text
    for pattern in ARTIFACT_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_dataset(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    required_columns = {"text", "label"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset must contain columns: {', '.join(sorted(required_columns))}. "
            f"Missing: {', '.join(sorted(missing))}."
        )

    records = data.loc[:, ["text", "label"]].dropna().copy()
    records["text"] = records["text"].astype(str).str.strip()
    records["label"] = records["label"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    records = records.loc[
        records["text"].ne("") & records["label"].notna()
    ].copy()

    # Clean publisher artifacts to eliminate wire shortcut bias
    records["text"] = records["text"].apply(clean_publisher_artifacts)
    records = records.loc[records["text"].ne("")].copy()

    records["normalized_text"] = (
        records["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    )
    if records.empty:
        raise ValueError("Dataset contains no non-empty rows with REAL/FAKE labels.")
    if set(records["label"]) != {"REAL", "FAKE"}:
        raise ValueError("Dataset must contain examples for both REAL and FAKE labels.")
    conflicting_labels = records.groupby("normalized_text")["label"].nunique().gt(1)
    if conflicting_labels.any():
        raise ValueError(
            "Duplicate article text has conflicting labels. Resolve those records before training."
        )
    records = records.drop_duplicates(subset="normalized_text", keep="first")
    texts = records["text"].tolist()
    labels = records["label"].tolist()
    if min(pd.Series(labels).value_counts()) < 10:
        raise ValueError(
            "At least ten unique articles per class are required for the "
            "stratified holdout and 5-fold model selection."
        )
    return texts, labels


def create_searches() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    def vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            stop_words="english",
            max_features=50000,
            sublinear_tf=True,
        )

    common_parameters: dict[str, list[Any]] = {}
    return {
        "TF-IDF + Logistic Regression": (
            Pipeline([
                ("tfidf", vectorizer()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            {**common_parameters, "classifier__C": [0.5, 2.0]},
        ),
        "TF-IDF + Linear SVM": (
            Pipeline([
                ("tfidf", vectorizer()),
                ("classifier", LinearSVC(random_state=RANDOM_STATE)),
            ]),
            {**common_parameters, "classifier__C": [0.5, 2.0]},
        ),
        "TF-IDF + Multinomial Naive Bayes": (
            Pipeline([("tfidf", vectorizer()), ("classifier", MultinomialNB())]),
            {**common_parameters, "classifier__alpha": [0.5, 1.0]},
        ),
    }


def evaluate_models(
    texts: list[str], labels: list[str]
) -> dict[str, Any]:
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    results: dict[str, Any] = {}
    selection_folds = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )
    for name, (pipeline, parameter_grid) in create_searches().items():
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            cv=selection_folds,
            scoring="accuracy",
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)

        cm = confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"]).tolist()

        res_dict = {
            "selection_cv_accuracy_mean": float(search.best_score_),
            "selection_cv_accuracy_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "best_parameters": search.best_params_,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "precision_macro": float(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "confusion_matrix": cm,
            "confusion_matrix_labels": ["FAKE", "REAL"],
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=["FAKE", "REAL"],
                output_dict=True,
                zero_division=0,
            ),
        }

        # Compute ROC-AUC if decision scores available
        best_pipe = search.best_estimator_
        if hasattr(best_pipe, "decision_function"):
            decisions = best_pipe.decision_function(x_test)
            y_binary = [1 if y == "REAL" else 0 for y in y_test]
            res_dict["roc_auc"] = float(roc_auc_score(y_binary, decisions))
        elif hasattr(best_pipe, "predict_proba"):
            probs = best_pipe.predict_proba(x_test)[:, 1]
            y_binary = [1 if y == "REAL" else 0 for y in y_test]
            res_dict["roc_auc"] = float(roc_auc_score(y_binary, probs))

        results[name] = res_dict
    return results


def train(data_path: Path, output_dir: Path) -> dict[str, Any]:
    data = pd.read_csv(data_path)
    texts, labels = normalize_dataset(data)
    metrics = evaluate_models(texts, labels)
    selected_name = max(
        metrics, key=lambda name: metrics[name]["selection_cv_accuracy_mean"]
    )

    final_pipeline, _ = create_searches()[selected_name]
    final_pipeline.set_params(**metrics[selected_name]["best_parameters"])
    final_pipeline.fit(texts, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": final_pipeline,
        "model_name": selected_name,
        "classes": ["FAKE", "REAL"],
        "training_rows": len(texts),
        "selection_metric": "mean accuracy across 3-fold CV on training data",
        "random_state": RANDOM_STATE,
        "cleaned_artifacts": True,
        "stop_words": "english",
    }
    joblib.dump(artifact, output_dir / "classifier.joblib")
    report = {
        "dataset": str(data_path),
        "training_rows": len(texts),
        "holdout_fraction": 0.2,
        "random_state": RANDOM_STATE,
        "selected_model": selected_name,
        "selection_metric": "mean accuracy across 3-fold CV on training data",
        "models": metrics,
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF news classifiers and save the best pipeline."
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="CSV file with text,label columns"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "artifacts"
    )
    args = parser.parse_args()
    report = train(args.data, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
