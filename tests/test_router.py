import unittest
from wikirag.retrieval.query_router import QueryRouter, QueryArchetype


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_classification_factual(self):
        q1 = "ริมุรุ เป็นเผ่าพันธุ์อะไร"
        self.assertEqual(self.router.classify(q1), QueryArchetype.FACTUAL)

        q2 = "What is Rimuru's species and rank?"
        self.assertEqual(self.router.classify(q2), QueryArchetype.FACTUAL)

    def test_classification_relational(self):
        q1 = "ริมุรุ มีความสัมพันธ์ยังไงกับเวลโดร่า"
        self.assertEqual(self.router.classify(q1), QueryArchetype.RELATIONAL)

        q2 = "What is the relationship between Rimuru and Milim?"
        self.assertEqual(self.router.classify(q2), QueryArchetype.RELATIONAL)

    def test_classification_comparative(self):
        q1 = "ริมุรุกับกี คริมสัน ใครเก่งกว่ากัน"
        self.assertEqual(self.router.classify(q1), QueryArchetype.COMPARATIVE)

        q2 = "Rimuru vs Guy Crimson who is stronger?"
        self.assertEqual(self.router.classify(q2), QueryArchetype.COMPARATIVE)

    def test_strategy_hints(self):
        hints = self.router.get_strategy_hints(QueryArchetype.FACTUAL)
        self.assertTrue(hints.get("prioritize_infobox"))


if __name__ == "__main__":
    unittest.main()
