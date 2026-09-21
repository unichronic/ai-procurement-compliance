import logging
from typing import List, Dict, Any, Optional
try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    GraphDatabase = None
    Driver = Any

from app.config import settings

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        self.driver: Optional[Driver] = None
        self.connected: bool = False
        self._init_driver()

    def _init_driver(self):
        if GraphDatabase is None:
            logger.warning("neo4j package is not installed. Neo4j graph functionality is disabled.")
            return

        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            self.connected = True
            logger.info("Successfully connected to Neo4j Graph Database.")
        except Exception as e:
            logger.warning(f"Neo4j connection failed ({settings.NEO4J_URI}): {e}. Neo4j operations will run in mock/disabled mode.")
            self.connected = False

    def close(self):
        if self.driver:
            self.driver.close()

    def create_constraints(self):
        """Create uniqueness constraints and indexes in Neo4j."""
        if not self.connected:
            return
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Standard) REQUIRE s.number IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:ProductCategory) REQUIRE c.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cs:CertificationScheme) REQUIRE cs.type IS UNIQUE;"
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Failed to execute Cypher constraint: {e}")

    def upsert_category(self, category_id: str, name: str):
        if not self.connected:
            return
        query = """
        MERGE (c:ProductCategory {id: $id})
        SET c.name = $name
        """
        with self.driver.session() as session:
            session.run(query, id=category_id, name=name)

    def upsert_standard(self, standard_number: str, title: str, status: str, category_id: Optional[str] = None):
        if not self.connected:
            return
        query = """
        MERGE (s:Standard {number: $number})
        SET s.title = $title, s.status = $status
        """
        with self.driver.session() as session:
            session.run(query, number=standard_number, title=title, status=status)
            if category_id:
                session.run("""
                MATCH (s:Standard {number: $number})
                MATCH (c:ProductCategory {id: $cat_id})
                MERGE (s)-[:BELONGS_TO]->(c)
                """, number=standard_number, cat_id=category_id)

    # Closed vocabulary of relationship types. Anything not here is refused
    # rather than escaped, because escaping an identifier into a Cypher string
    # is not something to get subtly wrong.
    ALLOWED_REL_TYPES = frozenset({
        "NORMATIVE_REFERENCE", "TEST_METHOD", "TERMINOLOGY", "SAFETY",
        "INSTALLATION", "RELATED_PRODUCT", "SUPERSEDES", "SUPERSEDED_BY",
        "BELONGS_TO", "REQUIRES_CERTIFICATION", "AMENDED_BY", "REFERS_TO",
    })

    @classmethod
    def _validate_rel_type(cls, rel_type: str) -> str:
        normalised = (rel_type or "").strip().upper().replace(" ", "_").replace("-", "_")
        if normalised not in cls.ALLOWED_REL_TYPES:
            raise ValueError(
                f"Refusing unknown relationship type {rel_type!r}. "
                f"Allowed: {sorted(cls.ALLOWED_REL_TYPES)}"
            )
        return normalised

    def upsert_relationship(self, source_number: str, target_number: str, rel_type: str, description: Optional[str] = None):
        if not self.connected:
            return
        # Cypher cannot parameterise a relationship type, so it has to be
        # interpolated -- which makes an allowlist the only safe option.
        # Replacing spaces and hyphens (the previous approach) leaves
        # backticks, braces and newlines intact, so a rel_type sourced from an
        # ingested document could close the pattern and append arbitrary
        # Cypher. Relationship types are a closed vocabulary here, so anything
        # outside it is a bug or an attack and is rejected either way.
        safe_rel_type = self._validate_rel_type(rel_type)
        query = f"""
        MERGE (src:Standard {{number: $source}})
        MERGE (tgt:Standard {{number: $target}})
        MERGE (src)-[r:`{safe_rel_type}`]->(tgt)
        SET r.description = $description
        """
        with self.driver.session() as session:
            session.run(query, source=source_number, target=target_number, description=description or "")

    def upsert_certification(self, category_id: str, standard_number: Optional[str], scheme_type: str):
        if not self.connected:
            return
        with self.driver.session() as session:
            session.run("""
            MERGE (cs:CertificationScheme {type: $scheme})
            """, scheme=scheme_type)

            if standard_number:
                session.run("""
                MATCH (s:Standard {number: $number})
                MATCH (cs:CertificationScheme {type: $scheme})
                MERGE (s)-[:REQUIRES_CERTIFICATION]->(cs)
                """, number=standard_number, scheme=scheme_type)

    def get_stats(self) -> Dict[str, Any]:
        if not self.connected:
            return {"nodes": 0, "relationships": 0, "status": "disconnected"}
        try:
            with self.driver.session() as session:
                node_res = session.run("MATCH (n) RETURN count(n) AS node_count").single()
                rel_res = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count").single()
                return {
                    "nodes": node_res["node_count"] if node_res else 0,
                    "relationships": rel_res["rel_count"] if rel_res else 0,
                    "status": "connected"
                }
        except Exception as e:
            return {"nodes": 0, "relationships": 0, "status": f"error: {str(e)}"}

    def clear_all(self):
        if not self.connected:
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

_client_singleton: Optional["Neo4jClient"] = None


def get_neo4j_client() -> "Neo4jClient":
    """Lazily construct the shared client.

    Previously this module instantiated a client at import time, so merely
    importing it -- during test collection, or a CLI that never touches the
    graph -- opened a network connection and paid its timeout.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = Neo4jClient()
    return _client_singleton
