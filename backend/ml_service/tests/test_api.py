import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from ml_service import main


class PredictionApiTests(unittest.TestCase):
    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            main.PredictionRequest(text="   ")

    def test_excessively_long_text_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            main.PredictionRequest(text="x" * (main.MAX_TEXT_LENGTH + 1))

    def test_prediction_reports_unavailable_model(self) -> None:
        with patch.object(main, "model_artifact", None):
            with self.assertRaises(HTTPException) as error:
                main.predict(main.PredictionRequest(text="A complete article headline"))
        self.assertEqual(error.exception.status_code, 503)

    def test_prediction_returns_model_probability_when_supported(self) -> None:
        class Pipeline:
            classes_ = ["FAKE", "REAL"]

            def predict(self, texts):
                return ["REAL"]

            def predict_proba(self, texts):
                return [[0.2, 0.8]]

        with patch.object(
            main,
            "model_artifact",
            {"pipeline": Pipeline(), "model_name": "test model"},
        ):
            result = main.predict(main.PredictionRequest(text="A complete article headline"))
        self.assertEqual(result["label"], "REAL")
        self.assertEqual(result["model_name"], "test model")
        self.assertEqual(result["probabilities"], {"FAKE": 0.2, "REAL": 0.8})

    def test_prediction_computes_probability_from_decision_function(self) -> None:
        class DecisionPipeline:
            classes_ = ["FAKE", "REAL"]

            def predict(self, texts):
                return ["REAL"]

            def decision_function(self, texts):
                # Positive decision = REAL leaning
                return [2.0]

        with patch.object(
            main,
            "model_artifact",
            {"pipeline": DecisionPipeline(), "model_name": "TF-IDF + Linear SVM"},
        ):
            result = main.predict(main.PredictionRequest(text="A complete article headline"))
        self.assertEqual(result["label"], "REAL")
        self.assertIn("probabilities", result)
        self.assertGreater(result["probabilities"]["REAL"], 0.8)
        self.assertLess(result["probabilities"]["FAKE"], 0.2)

    def test_analyze_endpoint_returns_complete_signals_and_breakdown(self) -> None:
        class DummyPipeline:
            classes_ = ["FAKE", "REAL"]

            def predict(self, texts):
                return ["FAKE"]

            def predict_proba(self, texts):
                return [[0.85, 0.15]]

        with patch.object(
            main,
            "model_artifact",
            {"pipeline": DummyPipeline(), "model_name": "Dummy ML"},
        ):
            response = main.analyze(
                main.AnalysisRequest(
                    text="SHOCKING: Secret conspiracy exposed in urgent leak!!",
                    url="https://reuters.com/world/article-123",
                )
            )

        self.assertEqual(response["finalLabel"], "FAKE")
        self.assertEqual(response["finalScore"], 85)
        self.assertIsNotNone(response["credibilityBreakdown"])
        self.assertEqual(response["credibilityBreakdown"]["contentScore"], 85)
        self.assertEqual(response["credibilityBreakdown"]["httpsScore"], 100)
        self.assertTrue(response["urlFeatures"]["isTrusted"])
        self.assertGreater(response["mlResult"]["textMetrics"]["sensationalism_score"], 20)

    def test_health_endpoint(self) -> None:
        health_data = main.health()
        self.assertEqual(health_data["status"], "ok")
        self.assertIn("model_loaded", health_data)
        self.assertIn("algorithms_available", health_data)

    def test_analysis_request_rejects_invalid_url_protocol(self) -> None:
        with self.assertRaises(ValidationError):
            main.AnalysisRequest(url="ftp://suspicious.site/data", text="valid text")
        with self.assertRaises(ValidationError):
            main.AnalysisRequest(url="file:///etc/passwd", text="valid text")

    def test_analysis_request_rejects_excessively_long_text(self) -> None:
        with self.assertRaises(ValidationError):
            main.AnalysisRequest(text="x" * (main.MAX_TEXT_LENGTH + 1))

    def test_models_metadata_endpoint(self) -> None:
        models_data = main.get_models()
        self.assertIn("model_loaded", models_data)
        self.assertIn("dataset_rows", models_data)


if __name__ == "__main__":
    unittest.main()
