import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "isot_fake_news.csv"
ARTIFACT_PATH = BASE_DIR / "artifacts" / "classifier.joblib"
REPORT_PATH = BASE_DIR / "artifacts" / "evaluation_report.json"
RANDOM_STATE = 42

LABEL_ALIASES = {
    "fake": "FAKE",
    "false": "FAKE",
    "real": "REAL",
    "true": "REAL",
}


def run_audit():
    print("=" * 70)
    print("           TRUTHLENS ML PIPELINE & DATASET AUDIT REPORT")
    print("=" * 70)

    # 1. Dataset Loading & Integrity Check
    print("\n--- 1. DATASET INTEGRITY & CLASS BALANCE ---")
    if not DATA_PATH.is_file():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return

    raw_df = pd.read_csv(DATA_PATH)
    total_raw = len(raw_df)
    print(f"Total raw rows: {total_raw:,}")
    print(f"Columns: {list(raw_df.columns)}")

    # Nulls
    null_texts = raw_df["text"].isna().sum()
    null_labels = raw_df["label"].isna().sum()
    print(f"Missing text count: {null_texts} ({null_texts/total_raw*100:.2f}%)")
    print(f"Missing label count: {null_labels} ({null_labels/total_raw*100:.2f}%)")

    # Clean & normalize as train.py does
    df = raw_df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    df = df[df["text"].ne("") & df["label"].notna()].copy()
    df["norm_text"] = df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)

    # Duplicates & Conflicts
    conflicts = df.groupby("norm_text")["label"].nunique().gt(1)
    conflict_count = int(conflicts.sum())
    print(f"Conflicting duplicate text records: {conflict_count}")

    unique_df = df.drop_duplicates(subset="norm_text", keep="first").copy()
    total_unique = len(unique_df)
    print(f"Total unique usable rows after deduplication: {total_unique:,}")

    # Class Distribution
    class_counts = unique_df["label"].value_counts()
    print("\nClass distribution (clean & deduplicated):")
    for lbl, cnt in class_counts.items():
        print(f"  - {lbl}: {cnt:,} ({cnt/total_unique*100:.2f}%)")

    # 2. Text Length Distribution
    print("\n--- 2. TEXT LENGTH & STATISTICAL DISTRIBUTION ---")
    unique_df["char_len"] = unique_df["text"].str.len()
    unique_df["word_len"] = unique_df["text"].str.split().str.len()

    for lbl in ["REAL", "FAKE"]:
        sub = unique_df[unique_df["label"] == lbl]
        print(f"\nStats for {lbl} (words):")
        print(f"  Mean: {sub['word_len'].mean():.1f} words")
        print(f"  Median: {sub['word_len'].median():.1f} words")
        print(f"  Std: {sub['word_len'].std():.1f} words")
        print(f"  Min: {sub['word_len'].min()} words | Max: {sub['word_len'].max()} words")
        print(f"  < 20 words (headline length): {(sub['word_len'] < 20).sum():,} articles ({(sub['word_len'] < 20).mean()*100:.2f}%)")

    # 3. Dataset Artifacts & Leakage Detection (The Reuters Byline issue)
    print("\n--- 3. DATASET ARTIFACT & SPURIOUS CORRELATION AUDIT ---")
    reuters_pattern = re.compile(r"\b(reuters)\b", re.IGNORECASE)
    byline_pattern = re.compile(r"^\s*([A-Z\s,]+)\s*\((reuters)\)\s*[-—]", re.IGNORECASE)

    real_has_reuters = unique_df[unique_df["label"] == "REAL"]["text"].apply(lambda t: bool(reuters_pattern.search(t))).mean()
    fake_has_reuters = unique_df[unique_df["label"] == "FAKE"]["text"].apply(lambda t: bool(reuters_pattern.search(t))).mean()

    real_has_byline = unique_df[unique_df["label"] == "REAL"]["text"].apply(lambda t: bool(byline_pattern.search(t))).mean()
    fake_has_byline = unique_df[unique_df["label"] == "FAKE"]["text"].apply(lambda t: bool(byline_pattern.search(t))).mean()

    print(f"Articles containing the word 'reuters':")
    print(f"  - In REAL articles: {real_has_reuters*100:.2f}%")
    print(f"  - In FAKE articles: {fake_has_reuters*100:.2f}%")
    print(f"Articles starting with wire service bylines (e.g. 'WASHINGTON (Reuters) -'):")
    print(f"  - In REAL articles: {real_has_byline*100:.2f}%")
    print(f"  - In FAKE articles: {fake_has_byline*100:.2f}%")

    # 4. Existing Model Audit on Stratified Holdout Test Split
    print("\n--- 4. EXISTING MODEL EVALUATION ON STRATIFIED HOLDOUT TEST SET ---")
    if not ARTIFACT_PATH.is_file():
        print(f"ERROR: Model artifact not found at {ARTIFACT_PATH}")
        return

    artifact = joblib.load(ARTIFACT_PATH)
    pipeline = artifact["pipeline"]
    model_name = artifact.get("model_name", "Unknown")
    print(f"Loaded deployed model artifact: '{model_name}'")
    print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    print(f"Classes: {pipeline.classes_}")

    texts = unique_df["text"].tolist()
    labels = unique_df["label"].tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Test partition size: {len(x_test):,} (stratified 20%)")

    # In-domain test predictions
    y_pred = pipeline.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_wt = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])

    tn, fp, fn, tp = cm.ravel()  # FAKE is negative (0), REAL is positive (1)
    fpr_real = fp / (fp + tn) if (fp + tn) > 0 else 0  # Actual FAKE predicted as REAL
    fnr_real = fn / (fn + tp) if (fn + tp) > 0 else 0  # Actual REAL predicted as FAKE

    print(f"Overall Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy:         {bacc*100:.2f}%")
    print(f"Macro Precision:           {prec_macro*100:.2f}%")
    print(f"Macro Recall:              {rec_macro*100:.2f}%")
    print(f"Macro F1-Score:            {f1_mac*100:.2f}%")
    print(f"Weighted F1-Score:         {f1_wt*100:.2f}%")
    print(f"Confusion Matrix (rows=actual, cols=predicted [FAKE, REAL]):")
    print(f"  [[{tn:5d} (TN: Actual FAKE -> Pred FAKE), {fp:5d} (FP: Actual FAKE -> Pred REAL)]")
    print(f"   [{fn:5d} (FN: Actual REAL -> Pred FAKE), {tp:5d} (TP: Actual REAL -> Pred REAL)]]")
    print(f"False Positive Rate (for REAL): {fpr_real*100:.2f}%")
    print(f"False Negative Rate (for REAL): {fnr_real*100:.2f}%")

    pred_counts = pd.Series(y_pred).value_counts()
    print("In-domain Test Predicted Class Distribution:")
    for lbl, count in pred_counts.items():
        print(f"  - Predicted {lbl}: {count:,} ({count/len(y_pred)*100:.2f}%)")

    # Decision scores / ROC-AUC if applicable
    if hasattr(pipeline, "decision_function"):
        decisions = pipeline.decision_function(x_test)
        # Class 1 is REAL
        y_binary = [1 if y == "REAL" else 0 for y in y_test]
        auc = roc_auc_score(y_binary, decisions)
        print(f"ROC-AUC Score:             {auc:.4f}")

    # 5. Top Feature Weights & Bias Analysis
    print("\n--- 5. TOP FEATURE WEIGHTS IN EXISTING MODEL ---")
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    # Most positive (leans REAL)
    top_real_idx = np.argsort(coefs)[-15:][::-1]
    # Most negative (leans FAKE)
    top_fake_idx = np.argsort(coefs)[:15]

    print("Top 15 features predicting REAL (Positive coefficients):")
    for idx in top_real_idx:
        print(f"  {feature_names[idx]:<25} : {coefs[idx]:+.4f}")

    print("\nTop 15 features predicting FAKE (Negative coefficients):")
    for idx in top_fake_idx:
        print(f"  {feature_names[idx]:<25} : {coefs[idx]:+.4f}")

    # 6. Sensitivity / Adversarial / Distribution Shift Audit
    print("\n--- 6. SPURIOUS CORRELATION & BIAS TESTING ---")
    test_reals = [x for x, y in zip(x_test, y_test) if y == "REAL"][:100]

    # Strip 'reuters' from genuine real articles
    reals_without_reuters = [re.sub(r"(?i)\b(washington\s*\()?reuters(\))?(\s*[-—])?", "", t).strip() for t in test_reals]
    preds_orig = pipeline.predict(test_reals)
    preds_stripped = pipeline.predict(reals_without_reuters)

    orig_real_acc = (preds_orig == "REAL").mean()
    stripped_real_acc = (preds_stripped == "REAL").mean()
    flipped_to_fake = ((preds_orig == "REAL") & (preds_stripped == "FAKE")).sum()

    print(f"Sample of 100 actual REAL test articles:")
    print(f"  Accuracy WITH 'Reuters' bylines intact: {orig_real_acc*100:.1f}%")
    print(f"  Accuracy WHEN 'Reuters' byline is removed: {stripped_real_acc*100:.1f}%")
    print(f"  Articles that flipped from REAL -> FAKE purely due to removing 'Reuters': {flipped_to_fake} / 100 ({flipped_to_fake}%)!")

    # Now test adding "WASHINGTON (Reuters) -" to obvious FAKE text
    fake_samples = [
        "Shocking secret proof exposed: alien lizard people secretly run the world government from hidden underground bunker!",
        "Miracle cure secret revealed: drinking raw lemon bleach cures all cancers overnight, doctors are terrified!",
        "Celebrity caught in massive satanic ritual scandal, mainstream media is completely covering it up!",
        "Breaking: Bill Gates admits microchips in vaccines during private closed door conference leaked audio!",
    ]
    fake_preds_plain = pipeline.predict(fake_samples)
    fake_samples_with_reuters = ["WASHINGTON (Reuters) - " + s for s in fake_samples]
    fake_preds_with_reuters = pipeline.predict(fake_samples_with_reuters)

    print("\nAdversarial Byline Injection Test (Adding 'WASHINGTON (Reuters) -' to Fake News):")
    for s, p1, p2 in zip(fake_samples, fake_preds_plain, fake_preds_with_reuters):
        print(f"  Original Fake: [{p1}] -> With '(Reuters) -': [{p2}]")
        print(f"    Text: '{s[:65]}...'")

    # 7. Real-world Generalization on Independent Modern News Sources
    print("\n--- 7. INDEPENDENT REAL-WORLD NEWS GENERALIZATION AUDIT ---")
    independent_real_news = [
        "The Federal Reserve held interest rates steady on Wednesday following a two-day policy meeting, citing sustained economic growth and solid job gains.",
        "Scientists at the James Webb Space Telescope announced the discovery of six massive candidate galaxies that formed much earlier in the universe than expected.",
        "The European Union reached a landmark agreement on artificial intelligence regulations, establishing comprehensive legal guardrails for high-risk AI models.",
        "The Supreme Court heard oral arguments on Tuesday regarding federal regulatory power and environmental emissions standards.",
        "Prime Minister Rishi Sunak addressed parliament today on economic growth, inflation targets, and NHS workforce investments.",
        "Japan successfully launched a new flagship rocket carrying an earth observation satellite from the Tanegashima Space Center.",
        "The World Health Organization reported a steady decline in regional cholera cases following emergency vaccination and clean water initiatives.",
        "NASA and the Canadian Space Agency announced the four astronauts assigned to fly around the moon on the Artemis II mission.",
        "The Bank of England raised interest rates by a quarter point in an ongoing effort to return inflation to its two percent target.",
        "Doctors at Massachusetts General Hospital performed the first successful transplant of a genetically modified pig kidney into a human patient."
    ]

    real_preds = pipeline.predict(independent_real_news)
    pred_real_count = (real_preds == "REAL").sum()
    print(f"Test on 10 authentic, modern non-Reuters real news articles:")
    for news, pred in zip(independent_real_news, real_preds):
        print(f"  [{pred}] {news[:70]}...")
    print(f"-> Classified as REAL: {pred_real_count}/10 ({pred_real_count*10:.0f}%)")
    print(f"-> Classified as FAKE: {10 - pred_real_count}/10 ({(10 - pred_real_count)*10:.0f}%)")

    # 8. Short Headlines Test
    print("\n--- 8. SHORT HEADLINE AUDIT ---")
    real_headlines = [
        "Senate passes bipartisan infrastructure bill",
        "Stock market closes higher after inflation data",
        "NASA reveals stunning new images of Jupiter",
        "WHO warns of rising heatwaves in southern Europe",
        "Unemployment rate falls to 3.7 percent in November",
    ]
    headline_preds = pipeline.predict(real_headlines)
    print("Test on 5 authentic real headlines:")
    for h, p in zip(real_headlines, headline_preds):
        print(f"  [{p}] {h}")


if __name__ == "__main__":
    run_audit()
