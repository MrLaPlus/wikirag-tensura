import unittest
from wikirag.eval.metrics import compute_hit_rate_at_k, compute_keyword_recall, compute_mrr


class TestEvalMetrics(unittest.TestCase):
    def test_hit_rate(self):
        retrieved = ["Milim Nava", "Rimuru Tempest", "Guy Crimson"]
        expected = ["Rimuru Tempest"]

        self.assertEqual(compute_hit_rate_at_k(retrieved, expected, k=1), 0.0)
        self.assertEqual(compute_hit_rate_at_k(retrieved, expected, k=2), 1.0)
        self.assertEqual(compute_hit_rate_at_k(retrieved, expected, k=5), 1.0)

    def test_mrr(self):
        retrieved = ["Milim Nava", "Rimuru Tempest", "Guy Crimson"]
        expected = ["Rimuru Tempest"]

        # Rimuru is at rank 2 -> MRR = 1/2 = 0.5
        self.assertEqual(compute_mrr(retrieved, expected), 0.5)

    def test_keyword_recall(self):
        answer = "Rimuru evolved into an Ultimate Slime and Demon Lord."
        keywords = ["Ultimate Slime", "Demon Lord", "True Dragon"]

        # 2 out of 3 match -> 2/3 = 0.6667
        score = compute_keyword_recall(answer, keywords)
        self.assertAlmostEqual(score, 2.0 / 3.0, places=3)


if __name__ == "__main__":
    unittest.main()
