from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

os.environ.setdefault("PRELOAD_PATENT_ASSETS", "0")
os.environ.setdefault("QUERY_RETRIEVAL_MODE", "tfidf")

from backend.app.main import app


class PatentApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["status"], {"ok", "degraded"})
        self.assertIn("runtime", payload)

    def test_cluster_dashboard(self) -> None:
        response = self.client.get("/api/dashboard/clusters")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"]["num_clusters"], 5)
        self.assertEqual(len(payload["clusters"]), 5)
        self.assertGreater(len(payload["points"]), 0)

    def test_novelty_endpoint(self) -> None:
        response = self.client.post(
            "/api/novelty/score",
            json={
                "title": "Adaptive wireless scheduling",
                "abstract": "A system that improves wireless inference scheduling.",
                "claim_text": "Selecting a schedule based on radio conditions.",
                "top_k": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["retrieval_mode"], {"semantic-faiss", "tfidf-fallback"})
        self.assertEqual(len(payload["matches"]), 3)
        self.assertIsNotNone(payload["novelty_score"])

    def test_rag_endpoint(self) -> None:
        response = self.client.post(
            "/api/rag/chat",
            json={
                "question": "Which patents discuss operator network scheduling?",
                "top_k": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["mode"], {"llm", "retrieval-only"})
        self.assertIn(payload["retrieval_mode"], {"semantic-faiss", "tfidf-fallback"})
        self.assertEqual(len(payload["retrieved_chunks"]), 3)
        self.assertTrue(payload["answer"])


if __name__ == "__main__":
    unittest.main()
