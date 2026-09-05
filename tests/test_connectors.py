import os
import unittest
from pathlib import Path
from wikirag.config import load_project_config
from wikirag.connectors.local_files import LocalFilesConnector


class TestConnectors(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/test_docs")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        with open(self.test_dir / "rimuru_lore.md", "w", encoding="utf-8") as f:
            f.write("# Rimuru Tempest\nRimuru is the founder of Tempest.")

    def tearDown(self):
        if (self.test_dir / "rimuru_lore.md").exists():
            os.remove(self.test_dir / "rimuru_lore.md")
        if self.test_dir.exists():
            os.rmdir(self.test_dir)

    def test_local_files_connector(self):
        cfg = load_project_config("projects/tensura.yaml")
        cfg.source.base_url = str(self.test_dir)
        connector = LocalFilesConnector(cfg)

        records = list(connector.crawl_all())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "Rimuru Lore")
        self.assertIn("founder of Tempest", records[0]["wikitext"])


if __name__ == "__main__":
    unittest.main()
