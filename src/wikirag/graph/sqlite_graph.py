import json
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from wikirag.graph.models import EntityNode, GraphSubgraph, RelationshipEdge
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteEntityGraph:
    """Production local Knowledge Graph engine backed by SQLite.
    
    Provides:
    - Zero external database ops (no Neo4j server process required)
    - Entities and Triples (Subject -> Predicate -> Object)
    - 1-hop and 2-hop BFS neighborhood expansion
    - Shortest path discovery between any two entities
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Entities table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT DEFAULT 'character',
                    attributes_json TEXT,
                    aliases_json TEXT
                )
                """
            )
            # Relationships table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_chunk_id TEXT,
                    properties_json TEXT,
                    UNIQUE(source, target, relation_type)
                )
                """
            )
            # Indexes for fast bidirectional traversal
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type)")
            conn.commit()

    def upsert_entity(self, node: EntityNode) -> None:
        """Inserts or updates an entity node."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO entities (id, name, entity_type, attributes_json, aliases_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    entity_type=excluded.entity_type,
                    attributes_json=excluded.attributes_json,
                    aliases_json=excluded.aliases_json
                """,
                (
                    node.id,
                    node.name,
                    node.entity_type,
                    json.dumps(node.attributes, ensure_ascii=False),
                    json.dumps(node.aliases, ensure_ascii=False),
                ),
            )
            conn.commit()

    def add_relationship(self, edge: RelationshipEdge) -> bool:
        """Adds a directed relationship edge. Returns True if new, False if existed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO relationships (source, target, relation_type, confidence, source_chunk_id, properties_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, target, relation_type) DO UPDATE SET
                        confidence=excluded.confidence,
                        source_chunk_id=excluded.source_chunk_id,
                        properties_json=excluded.properties_json
                    """,
                    (
                        edge.source,
                        edge.target,
                        edge.relation_type,
                        edge.confidence,
                        edge.source_chunk_id,
                        json.dumps(edge.properties, ensure_ascii=False),
                    ),
                )
                conn.commit()
                return True
            except Exception as e:
                logger.debug(f"Edge insert error: {e}")
                return False

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Fetches a single entity node by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, entity_type, attributes_json, aliases_json FROM entities WHERE id = ?",
                (entity_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return EntityNode(
                id=row[0],
                name=row[1],
                entity_type=row[2],
                attributes=json.loads(row[3] or "{}"),
                aliases=json.loads(row[4] or "[]"),
            )

    def get_subgraph(self, entity_id: str, max_depth: int = 2, max_nodes: int = 40, relation_type: Optional[str] = None) -> GraphSubgraph:
        """Performs BFS graph traversal up to max_depth starting from entity_id."""
        visited_nodes: Set[str] = {entity_id}
        queue = deque([(entity_id, 0)])
        edges: List[RelationshipEdge] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            while queue and len(visited_nodes) < max_nodes:
                curr_node, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                # Query outgoing and incoming edges
                relation_sql = " AND relation_type = ?" if relation_type else ""
                params = (curr_node, curr_node, relation_type) if relation_type else (curr_node, curr_node)
                cursor.execute(
                    f"""SELECT id, source, target, relation_type, confidence, source_chunk_id, properties_json
                    FROM relationships WHERE (source = ? OR target = ?){relation_sql} LIMIT 30""",
                    params,
                )
                rows = cursor.fetchall()

                for r in rows:
                    e_id, src, tgt, r_type, conf, ch_id, props = r
                    edge_key = (src, tgt, r_type)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            RelationshipEdge(
                                id=e_id,
                                source=src,
                                target=tgt,
                                relation_type=r_type,
                                confidence=conf,
                                source_chunk_id=ch_id,
                                properties=json.loads(props or "{}"),
                            )
                        )

                    neighbor = tgt if src == curr_node else src
                    if neighbor not in visited_nodes and len(visited_nodes) < max_nodes:
                        visited_nodes.add(neighbor)
                        queue.append((neighbor, depth + 1))

            # Fetch node details for all visited entities
            nodes: List[EntityNode] = []
            if visited_nodes:
                placeholders = ",".join("?" for _ in visited_nodes)
                cursor.execute(
                    f"SELECT id, name, entity_type, attributes_json, aliases_json FROM entities WHERE id IN ({placeholders})",
                    list(visited_nodes),
                )
                for r in cursor.fetchall():
                    nodes.append(
                        EntityNode(
                            id=r[0],
                            name=r[1],
                            entity_type=r[2],
                            attributes=json.loads(r[3] or "{}"),
                            aliases=json.loads(r[4] or "[]"),
                        )
                    )

        return GraphSubgraph(nodes=nodes, edges=edges)

    def find_shortest_path(self, source: str, target: str) -> List[RelationshipEdge]:
        """Finds the shortest relational path connecting source and target."""
        if source == target:
            return []

        queue = deque([[source]])
        visited = {source}
        edge_map: Dict[Tuple[str, str], RelationshipEdge] = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            while queue:
                path = queue.popleft()
                curr = path[-1]

                if curr == target:
                    # Reconstruct edge list
                    result_edges = []
                    for i in range(len(path) - 1):
                        pair = (path[i], path[i + 1])
                        if pair in edge_map:
                            result_edges.append(edge_map[pair])
                    return result_edges

                cursor.execute(
                    """
                    SELECT id, source, target, relation_type, confidence, source_chunk_id, properties_json
                    FROM relationships
                    WHERE source = ? OR target = ?
                    """,
                    (curr, curr),
                )
                for r in cursor.fetchall():
                    e_id, s, t, r_type, conf, ch_id, props = r
                    neighbor = t if s == curr else s
                    if neighbor not in visited:
                        visited.add(neighbor)
                        edge_obj = RelationshipEdge(
                            id=e_id,
                            source=s,
                            target=t,
                            relation_type=r_type,
                            confidence=conf,
                            source_chunk_id=ch_id,
                            properties=json.loads(props or "{}"),
                        )
                        edge_map[(curr, neighbor)] = edge_obj
                        new_path = list(path)
                        new_path.append(neighbor)
                        queue.append(new_path)

        return []

    def get_overview_graph(self, limit_nodes: int = 50, entity_type: Optional[str] = None, relation_type: Optional[str] = None) -> GraphSubgraph:
        """Returns the top connected subgraph for overall platform visualization."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Fetch entities with most relationships
            type_sql = " WHERE entity_type = ?" if entity_type else ""
            type_params = (entity_type, limit_nodes) if entity_type else (limit_nodes,)
            cursor.execute(
                f"""SELECT e.id, e.name, e.entity_type, e.attributes_json, e.aliases_json
                FROM entities e
                LEFT JOIN (
                    SELECT source AS entity_id FROM relationships
                    UNION ALL
                    SELECT target AS entity_id FROM relationships
                ) degree ON degree.entity_id = e.id
                {type_sql}
                GROUP BY e.id, e.name, e.entity_type, e.attributes_json, e.aliases_json
                ORDER BY COUNT(degree.entity_id) DESC, e.name COLLATE NOCASE
                LIMIT ?""",
                type_params,
            )
            node_rows = cursor.fetchall()
            node_ids = {r[0] for r in node_rows}
            nodes = [
                EntityNode(
                    id=r[0],
                    name=r[1],
                    entity_type=r[2],
                    attributes=json.loads(r[3] or "{}"),
                    aliases=json.loads(r[4] or "[]"),
                )
                for r in node_rows
            ]

            if not node_ids:
                return GraphSubgraph(nodes=[], edges=[])

            placeholders = ",".join("?" for _ in node_ids)
            relation_sql = " AND relation_type = ?" if relation_type else ""
            relation_params = list(node_ids) + list(node_ids) + ([relation_type] if relation_type else [])
            cursor.execute(
                f"""
                SELECT id, source, target, relation_type, confidence, source_chunk_id, properties_json
                FROM relationships
                WHERE source IN ({placeholders}) AND target IN ({placeholders})
                {relation_sql}
                LIMIT 100
                """,
                relation_params,
            )
            edges = [
                RelationshipEdge(
                    id=r[0],
                    source=r[1],
                    target=r[2],
                    relation_type=r[3],
                    confidence=r[4],
                    source_chunk_id=r[5],
                    properties=json.loads(r[6] or "{}"),
                )
                for r in cursor.fetchall()
            ]

        return GraphSubgraph(nodes=nodes, edges=edges)
