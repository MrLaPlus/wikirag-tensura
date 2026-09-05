import os
import tempfile
import unittest
from pathlib import Path
from wikirag.graph.builder import GraphBuilder
from wikirag.graph.models import EntityNode, RelationshipEdge
from wikirag.graph.sqlite_graph import SQLiteEntityGraph
from wikirag.parser.models import InfoboxData, ParsedPage, SectionData


class TestEntityGraph(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        self.db_path = self.temp_file.name
        self.graph = SQLiteEntityGraph(self.db_path)

    def tearDown(self):
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception:
            pass


    def test_upsert_and_get_entity(self):
        node = EntityNode(
            id="Rimuru Tempest",
            name="Rimuru Tempest",
            entity_type="character",
            attributes={"species": "Ultimate Slime"},
            aliases=["Slime", "Satoru Mikami"],
        )
        self.graph.upsert_entity(node)

        fetched = self.graph.get_entity("Rimuru Tempest")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, "Rimuru Tempest")
        self.assertEqual(fetched.attributes.get("species"), "Ultimate Slime")
        self.assertIn("Slime", fetched.aliases)

    def test_bfs_subgraph_and_path(self):
        # Insert entities
        self.graph.upsert_entity(EntityNode(id="Rimuru", name="Rimuru"))
        self.graph.upsert_entity(EntityNode(id="Veldora", name="Veldora"))
        self.graph.upsert_entity(EntityNode(id="Ifrit", name="Ifrit"))

        # Insert relationships: Rimuru <-> Veldora <-> Ifrit
        self.graph.add_relationship(
            RelationshipEdge(source="Rimuru", target="Veldora", relation_type="friend")
        )
        self.graph.add_relationship(
            RelationshipEdge(source="Veldora", target="Ifrit", relation_type="mentor")
        )

        # 1-hop from Rimuru
        subgraph = self.graph.get_subgraph("Rimuru", max_depth=1)
        node_ids = {n.id for n in subgraph.nodes}
        self.assertIn("Rimuru", node_ids)
        self.assertIn("Veldora", node_ids)
        self.assertNotIn("Ifrit", node_ids)

        # 2-hop from Rimuru
        subgraph2 = self.graph.get_subgraph("Rimuru", max_depth=2)
        node_ids2 = {n.id for n in subgraph2.nodes}
        self.assertIn("Ifrit", node_ids2)

        # Shortest path
        path = self.graph.find_shortest_path("Rimuru", "Ifrit")
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0].source, "Rimuru")
        self.assertEqual(path[1].target, "Ifrit")

    def test_graph_builder(self):
        builder = GraphBuilder(self.graph)
        page = ParsedPage(
            page_id=1,
            title="Rimuru Tempest",
            canonical_url="https://tensura.fandom.com/wiki/Rimuru_Tempest",
            categories=["Characters", "Demon Lords"],
            aliases=["Slime"],
            infobox=InfoboxData(
                name="Rimuru Tempest",
                species="Ultimate Slime",
                titles=["Demon Lord", "Chaos Creator"],
                raw_fields={"subordinates": "Benimaru, Shion"},
            ),

            lead_section="Rimuru is friends with [[Veldora Tempest]].",
            sections=[],
            wiki_links=["Veldora Tempest"],
            fetched_at=1700000000.0,
        )

        builder.process_page(page)

        # Verify entity created
        node = self.graph.get_entity("Rimuru Tempest")
        self.assertIsNotNone(node)

        # Verify species edge
        subgraph = self.graph.get_subgraph("Rimuru Tempest", max_depth=1)
        rel_types = {e.relation_type for e in subgraph.edges}
        self.assertIn("species_of", rel_types)
        self.assertIn("has_title", rel_types)
        self.assertIn("subordinate_of", rel_types)


if __name__ == "__main__":
    unittest.main()
