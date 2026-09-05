import unittest
from fastapi.testclient import TestClient
from wikirag.api.app import app


class TestFastAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_root_web_ui(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("WikiRAG", resp.text)
        self.assertIn("root", resp.text)


    def test_categories_endpoint(self):
        resp = self.client.get("/api/categories")
        self.assertEqual(resp.status_code, 200)
        cats = resp.json()
        self.assertIn("Characters", cats)
        self.assertIn("Demon Lords", cats)

    def test_projects_endpoint(self):
        resp = self.client.get("/api/projects")
        self.assertEqual(resp.status_code, 200)
        projects = resp.json()
        names = [p["name"] for p in projects]
        self.assertIn("tensura", names)


if __name__ == "__main__":
    unittest.main()
