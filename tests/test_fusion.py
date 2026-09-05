import unittest
from wikirag.retrieval.fusion import reciprocal_rank_fusion


class TestRRF(unittest.TestCase):
    def test_rrf_scoring_and_ranking(self):
        # Dense leg results
        list1 = [
            {"chunk_id": "doc_A", "score": 0.95},
            {"chunk_id": "doc_B", "score": 0.85},
            {"chunk_id": "doc_C", "score": 0.75},
        ]
        # Sparse BM25 leg results
        list2 = [
            {"chunk_id": "doc_B", "score": 12.5},
            {"chunk_id": "doc_D", "score": 10.1},
            {"chunk_id": "doc_A", "score": 8.0},
        ]

        # Merge with RRF (k=60)
        fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)

        self.assertEqual(len(fused), 3)

        # doc_B is rank 2 in list1 (1/62) and rank 1 in list2 (1/61) -> total = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # doc_A is rank 1 in list1 (1/61) and rank 3 in list2 (1/63) -> total = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
        # Therefore doc_B should rank #1, followed by doc_A
        self.assertEqual(fused[0]["chunk_id"], "doc_B")
        self.assertEqual(fused[1]["chunk_id"], "doc_A")
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])


if __name__ == "__main__":
    unittest.main()
