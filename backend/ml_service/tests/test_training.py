import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from ml_service.train import create_searches, evaluate_models, normalize_dataset


class TrainingTests(unittest.TestCase):
    def test_dataset_labels_are_normalized(self) -> None:
        texts, labels = normalize_dataset(
            pd.DataFrame(
                {
                    "text": [f"Report {index}" for index in range(10)]
                    + [f"Rumor {index}" for index in range(10)],
                    "label": ["real"] * 10 + ["fake"] * 10,
                }
            )
        )
        self.assertEqual(len(texts), 20)
        self.assertEqual(labels, ["REAL"] * 10 + ["FAKE"] * 10)

    def test_conflicting_duplicate_article_labels_are_rejected(self) -> None:
        data = pd.DataFrame(
            {
                "text": [f"Real report {i}" for i in range(10)]
                + [f"Fake report {i}" for i in range(9)]
                + ["  REAL   REPORT 0 "],
                "label": ["real"] * 10 + ["fake"] * 10,
            }
        )
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            normalize_dataset(data)

    def test_dataset_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "columns"):
            normalize_dataset(pd.DataFrame({"headline": ["article"]}))

    def test_all_requested_algorithms_are_configured(self) -> None:
        pipelines = create_searches()
        self.assertEqual(
            set(pipelines),
            {
                "TF-IDF + Logistic Regression",
                "TF-IDF + Linear SVM",
                "TF-IDF + Multinomial Naive Bayes",
            },
        )
        self.assertIsInstance(
            pipelines["TF-IDF + Logistic Regression"][0].named_steps["classifier"],
            LogisticRegression,
        )
        self.assertIsInstance(
            pipelines["TF-IDF + Linear SVM"][0].named_steps["classifier"],
            LinearSVC,
        )
        self.assertIsInstance(
            pipelines["TF-IDF + Multinomial Naive Bayes"][0].named_steps["classifier"],
            MultinomialNB,
        )

    def test_tuned_models_report_holdout_metrics(self) -> None:
        texts = [f"verified government report public data policy {i}" for i in range(30)]
        texts += [f"fabricated rumor shocking secret claim hoax {i}" for i in range(30)]
        labels = ["REAL"] * 30 + ["FAKE"] * 30

        results = evaluate_models(texts, labels)

        self.assertEqual(len(results), 3)
        for metrics in results.values():
            self.assertIn("selection_cv_accuracy_mean", metrics)
            self.assertIn("accuracy", metrics)
            self.assertIn("precision_macro", metrics)
            self.assertIn("recall_macro", metrics)
            self.assertIn("f1_macro", metrics)
            self.assertEqual(len(metrics["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
