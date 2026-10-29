import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "isot_fake_news.csv"
RANDOM_STATE = 42

LABEL_ALIASES = {
    "fake": "FAKE",
    "false": "FAKE",
    "real": "REAL",
    "true": "REAL",
}

# Artifacts commonly found in scraped blogs/wire feeds that cause label shortcuts
ARTIFACT_REGEXES = [
    re.compile(r"(?i)^\s*[A-Z\s,]+?\s*\((Reuters|AP|AFP)\)\s*[-—]\s*", re.UNICODE),
    re.compile(r"(?i)\b(Reuters|AP|AFP)\b", re.UNICODE),
    re.compile(r"(?i)\b(featured image|image via|read more|via)\b", re.UNICODE),
    re.compile(r"(?i)\b(21st century wire|american lookout|breitbart|infowars)\b", re.UNICODE),
    re.compile(r"https?://\S+|www\.\S+", re.UNICODE),
]


def clean_artifacts(text: str) -> str:
    cleaned = text
    for pattern in ARTIFACT_REGEXES:
        cleaned = pattern.sub(" ", cleaned)
    return " ".join(cleaned.split())


# Independent test suite of non-ISOT real and fake news to measure true out-of-domain generalization
OOD_TEST_SUITE = [
    # 10 Authentic Modern Real News (from science, tech, health, world, politics)
    ("The Federal Reserve held interest rates steady on Wednesday following a two-day policy meeting, citing sustained economic growth and solid job gains.", "REAL"),
    ("Scientists at the James Webb Space Telescope announced the discovery of six massive candidate galaxies that formed much earlier in the universe than expected.", "REAL"),
    ("The European Union reached a landmark agreement on artificial intelligence regulations, establishing comprehensive legal guardrails for high-risk AI models.", "REAL"),
    ("The Supreme Court heard oral arguments on Tuesday regarding federal regulatory power and environmental emissions standards.", "REAL"),
    ("Prime Minister Rishi Sunak addressed parliament today on economic growth, inflation targets, and NHS workforce investments.", "REAL"),
    ("Japan successfully launched a new flagship rocket carrying an earth observation satellite from the Tanegashima Space Center.", "REAL"),
    ("The World Health Organization reported a steady decline in regional cholera cases following emergency vaccination and clean water initiatives.", "REAL"),
    ("NASA and the Canadian Space Agency announced the four astronauts assigned to fly around the moon on the Artemis II mission.", "REAL"),
    ("The Bank of England raised interest rates by a quarter point in an ongoing effort to return inflation to its two percent target.", "REAL"),
    ("Doctors at Massachusetts General Hospital performed the first successful transplant of a genetically modified pig kidney into a human patient.", "REAL"),
    # 10 Obvious Fake / Hoax / Conspiratorial Stories
    ("Shocking secret proof exposed: alien lizard people secretly run the world government from hidden underground bunker!", "FAKE"),
    ("Miracle cure secret revealed: drinking raw lemon bleach cures all cancers overnight, doctors are terrified!", "FAKE"),
    ("Celebrity caught in massive satanic ritual scandal, mainstream media is completely covering it up!", "FAKE"),
    ("Breaking: Bill Gates admits microchips in vaccines during private closed door conference leaked audio!", "FAKE"),
    ("NASA insider admits the moon is an artificial hollow metallic structure created by ancient extraterrestrials!", "FAKE"),
    ("Government deploys weather control superweapons to cause intentional earthquakes across developing nations!", "FAKE"),
    ("Secret hidden pyramid discovered under Antarctic ice sheets contains alien flying saucers and giant skeletons!", "FAKE"),
    ("5G mobile network signals proven to transmit viruses directly into human blood cells through radio frequencies!", "FAKE"),
    ("Secret tribunal quietly arrests top world leaders and replaces them with lifelike silicone body doubles!", "FAKE"),
    ("Drinking water fluoridation confirmed to be a secret military mind control experiment to suppress psychic abilities!", "FAKE"),
]


def run_experiments():
    print("=" * 70)
    print("      EXPERIMENTS: COMPARING PREPROCESSING & ALGORITHMS")
    print("=" * 70)

    raw_df = pd.read_csv(DATA_PATH)
    df = raw_df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    df = df[df["text"].ne("") & df["label"].notna()].copy()
    df["norm_text"] = df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset="norm_text", keep="first").copy()

    # Create cleaned version without wire/publisher shortcut artifacts
    df["clean_text"] = df["text"].apply(clean_artifacts)

    labels = df["label"].tolist()
    raw_texts = df["text"].tolist()
    clean_texts = df["clean_text"].tolist()

    # Shared stratified split
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    y_train = [labels[i] for i in idx_train]
    y_test = [labels[i] for i in idx_test]

    # Model Pipelines to test
    models = {
        "1. Raw Text + TF-IDF (no stop_words) + Linear SVM (Current)": (
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("classifier", LinearSVC(C=2.0, random_state=RANDOM_STATE)),
            ]),
            False, # use raw text
        ),
        "2. Cleaned Text + TF-IDF (english stop_words) + Linear SVM": (
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000, sublinear_tf=True)),
                ("classifier", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
            ]),
            True, # use clean text
        ),
        "3. Cleaned Text + TF-IDF (english stop_words) + Calibrated Linear SVM": (
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000, sublinear_tf=True)),
                ("classifier", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_STATE), cv=3)),
            ]),
            True,
        ),
        "4. Cleaned Text + TF-IDF (english stop_words) + Logistic Regression": (
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000, sublinear_tf=True)),
                ("classifier", LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            True,
        ),
        "5. Cleaned Text + TF-IDF (english stop_words) + Multinomial Naive Bayes": (
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000, sublinear_tf=True)),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]),
            True,
        ),
        "6. Cleaned Text + TF-IDF (english stop_words) + Passive-Aggressive": (
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000, sublinear_tf=True)),
                ("classifier", CalibratedClassifierCV(PassiveAggressiveClassifier(C=0.5, random_state=RANDOM_STATE, max_iter=1000), cv=3)),
            ]),
            True,
        ),
    }

    results = []

    for name, (pipe, use_clean) in models.items():
        print(f"\nTraining & evaluating: {name}...")
        texts_train = [clean_texts[i] if use_clean else raw_texts[i] for i in idx_train]
        texts_test = [clean_texts[i] if use_clean else raw_texts[i] for i in idx_test]

        pipe.fit(texts_train, y_train)

        # In-domain held-out test
        preds_test = pipe.predict(texts_test)
        acc = accuracy_score(y_test, preds_test)
        bacc = balanced_accuracy_score(y_test, preds_test)
        prec_m = precision_score(y_test, preds_test, average="macro")
        rec_m = recall_score(y_test, preds_test, average="macro")
        f1_m = f1_score(y_test, preds_test, average="macro")

        # Independent OOD Test Suite (20 items: 10 real, 10 fake)
        ood_texts = [clean_artifacts(item[0]) if use_clean else item[0] for item in OOD_TEST_SUITE]
        ood_true = [item[1] for item in OOD_TEST_SUITE]
        ood_preds = pipe.predict(ood_texts)

        ood_acc = accuracy_score(ood_true, ood_preds)
        ood_real_rec = recall_score(ood_true, ood_preds, pos_label="REAL")
        ood_fake_rec = recall_score(ood_true, ood_preds, pos_label="FAKE")

        res_entry = {
            "name": name,
            "test_accuracy": acc,
            "test_balanced_acc": bacc,
            "test_macro_f1": f1_m,
            "ood_accuracy": ood_acc,
            "ood_real_recall": ood_real_rec,
            "ood_fake_recall": ood_fake_rec,
        }
        results.append(res_entry)
        print(f"  Held-out Test Acc: {acc*100:.2f}% | Balanced: {bacc*100:.2f}% | Macro-F1: {f1_m*100:.2f}%")
        print(f"  Out-of-Domain Accuracy: {ood_acc*100:.1f}% (Real Recall: {ood_real_rec*100:.1f}%, Fake Recall: {ood_fake_rec*100:.1f}%)")

    print("\n" + "=" * 70)
    print("                     SUMMARY COMPARISON TABLE")
    print("=" * 70)
    res_df = pd.DataFrame(results)
    for _, r in res_df.iterrows():
        print(f"{r['name'][:55]:<55} | Test: {r['test_accuracy']*100:.2f}% | OOD Acc: {r['ood_accuracy']*100:.1f}% | OOD Real: {r['ood_real_recall']*100:.1f}% | OOD Fake: {r['ood_fake_recall']*100:.1f}%")


if __name__ == "__main__":
    run_experiments()
