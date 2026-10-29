import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

from ml_service.experiment import DATA_PATH, RANDOM_STATE, clean_artifacts, OOD_TEST_SUITE

def test_ensemble():
    raw_df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    df = raw_df.drop_duplicates(subset=["text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(["REAL", "FAKE"])].copy()
    df["clean_text"] = df["text"].apply(clean_artifacts)

    # Subsample 15k rows for quick iteration
    df_sample = df.groupby("label", group_keys=False).apply(lambda x: x.sample(n=7500, random_state=RANDOM_STATE))

    X_train = df_sample["clean_text"].tolist()
    y_train = df_sample["label"].tolist()

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=30000, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)

    # Base models
    lr = LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_STATE)
    nb = MultinomialNB(alpha=0.5)
    svm = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_STATE), cv=3)

    lr.fit(X_train_vec, y_train)
    nb.fit(X_train_vec, y_train)
    svm.fit(X_train_vec, y_train)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("nb", nb), ("svm", svm)],
        voting="soft",
        weights=[1.0, 1.5, 1.0] # Give weight to NB which is less prone to publisher artifact overfit
    )
    ensemble.fit(X_train_vec, y_train)

    ood_texts = [clean_artifacts(item[0]) for item in OOD_TEST_SUITE]
    ood_true = [item[1] for item in OOD_TEST_SUITE]
    ood_vec = tfidf.transform(ood_texts)

    for name, clf in [("Logistic Regression", lr), ("Naive Bayes", nb), ("Calibrated SVM", svm), ("Weighted Ensemble", ensemble)]:
        probs = clf.predict_proba(ood_vec)
        # Class 0: FAKE, Class 1: REAL
        classes = list(clf.classes_)
        real_idx = classes.index("REAL")
        fake_idx = classes.index("FAKE")

        # Standard threshold 0.5
        preds_std = [classes[np.argmax(p)] for p in probs]
        real_rec = recall_score(ood_true, preds_std, pos_label="REAL")
        fake_rec = recall_score(ood_true, preds_std, pos_label="FAKE")
        acc = accuracy_score(ood_true, preds_std)
        print(f"{name:<22} (thresh=0.50): Acc={acc*100:.1f}% | Real Recall={real_rec*100:.1f}% | Fake Recall={fake_rec*100:.1f}%")

        # Tuned threshold: require P(FAKE) > 0.58 to call FAKE (reducing false-fake bias on short/neutral text)
        preds_tuned = ["FAKE" if p[fake_idx] > 0.58 else "REAL" for p in probs]
        real_rec_t = recall_score(ood_true, preds_tuned, pos_label="REAL")
        fake_rec_t = recall_score(ood_true, preds_tuned, pos_label="FAKE")
        acc_t = accuracy_score(ood_true, preds_tuned)
        print(f"{name:<22} (thresh=0.58): Acc={acc_t*100:.1f}% | Real Recall={real_rec_t*100:.1f}% | Fake Recall={fake_rec_t*100:.1f}%")
        print("-" * 65)

if __name__ == "__main__":
    test_ensemble()
