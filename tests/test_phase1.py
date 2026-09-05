import unittest
import json
import mwparserfromhell

from wikirag.config import load_project_config
from wikirag.utils.hashing import compute_sha256, compute_chunk_id
from wikirag.parser.models import ParsedPage, InfoboxData, SectionData
from wikirag.parser.infobox import clean_wikitext_value, extract_infobox
from wikirag.parser.wikitext import WikitextParser
from wikirag.chunking.chunker import SectionAwareChunker
from wikirag.retrieval.preprocessing import QueryPreprocessor


class TestPhase1(unittest.TestCase):
    def test_load_tensura_config(self):
        cfg = load_project_config("projects/tensura.yaml")
        self.assertEqual(cfg.project.name, "tensura")
        self.assertEqual(cfg.source.type, "mediawiki")
        self.assertIn("tensura.fandom.com", cfg.source.api_url)
        self.assertEqual(cfg.embedding.model_name, "BAAI/bge-m3")

    def test_hashing_determinism(self):
        text1 = "Rimuru Tempest is the leader of Jura Tempest Federation."
        h1 = compute_sha256(text1)
        h2 = compute_sha256(text1)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

        chunk_id = compute_chunk_id("Rimuru Tempest", "Appearance > Slime Form", 0, text1)
        self.assertIn("Rimuru_Tempest", chunk_id)
        self.assertIn("Appearance__Slime_Form", chunk_id)

    def test_clean_wikitext_value(self):
        raw_val = "[[True Dragon]] {{S|[[Ultimate slime]]}}<br>[[Demon slime]]"
        cleaned = clean_wikitext_value(raw_val)
        self.assertIn("True Dragon", cleaned)
        self.assertIn("(Ultimate slime)", cleaned)
        self.assertIn("Demon slime", cleaned)
        self.assertNotIn("[[", cleaned)
        self.assertNotIn("{{", cleaned)

    def test_extract_infobox(self):
        wikitext = """
        {{Infobox Character
        |kanji = リムル＝テンペスト
        |romaji = rimuru tenpesuto
        |titles = Chaos Creator<br>Great Demon Lord
        |species = [[True Dragon]]
        |gender = Genderless
        |status = Alive
        }}
        """
        code = mwparserfromhell.parse(wikitext)
        infobox = extract_infobox(code)
        self.assertIsNotNone(infobox)
        self.assertEqual(infobox.kanji, "リムル＝テンペスト")
        self.assertEqual(infobox.gender, "Genderless")
        self.assertIn("Chaos Creator", infobox.titles)
        self.assertIn("Great Demon Lord", infobox.titles)
        self.assertEqual(infobox.species, "True Dragon")

    def test_section_aware_chunking(self):
        cfg = load_project_config("projects/tensura.yaml")
        chunker = SectionAwareChunker(cfg.chunking, project_name="tensura")

        page = ParsedPage(
            page_id=2336,
            title="Rimuru Tempest",
            canonical_url="https://tensura.fandom.com/wiki/Rimuru_Tempest",
            categories=["Characters", "Slimes", "Demon Lords"],
            aliases=["Slime", "Satoru Mikami"],
            infobox=InfoboxData(
                name="Rimuru Tempest",
                species="Ultimate Slime",
                gender="Genderless",
            ),
            lead_section="Rimuru Tempest is the main protagonist of the series. They reincarnated as a slime in another world.",
            sections=[
                SectionData(
                    title="Slime Form",
                    level=2,
                    path="Appearance > Slime Form",
                    content="In normal form, Rimuru is a blue amorphous mass. It is very soft and bouncy.",
                )
            ],
            fetched_at=1700000000.0,
        )

        chunks = list(chunker.chunk_page(page))
        self.assertGreaterEqual(len(chunks), 2)

        infobox_chunks = [c for c in chunks if c.chunk_type == "infobox"]
        self.assertEqual(len(infobox_chunks), 1)
        self.assertIn("INFOBOX FACT SHEET", infobox_chunks[0].chunk_text)
        self.assertIn("Entity: Rimuru Tempest", infobox_chunks[0].chunk_text)

        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        self.assertGreaterEqual(len(text_chunks), 1)
        self.assertIn("Entity: Rimuru Tempest", text_chunks[0].chunk_text)
        self.assertIn("Aliases: Slime, Satoru Mikami", text_chunks[0].chunk_text)

    def test_query_preprocessor_thai_detection(self):
        qp = QueryPreprocessor()
        self.assertEqual(qp.detect_language("ริมุรุ เทมเพสต์ คือใคร"), "th")
        self.assertEqual(qp.detect_language("Who is Rimuru Tempest?"), "en")
        self.assertEqual(qp.detect_language("Rimuru สกิลอะไรบ้าง"), "th")


if __name__ == "__main__":
    unittest.main()
