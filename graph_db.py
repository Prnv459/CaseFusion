from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()


class CaseFusionGraph:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print("✅ Graph Service Connected to Neo4j")
        except Exception as e:
            print(f"❌ Graph Connection Failed: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    # 1. Ingest Case + Entities, Return Alerts
    def ingest_case_data(self, case_id, case_title, entities):
        """
        Saves case data and returns ALERTS if entities link to previous cases.

        entities: list of dicts like:
          { "type": "PHONE", "value": "9876543210" }
        """
        if not self.driver:
            print("❌ Graph DB Error: Driver not initialized.")
            return {
                "status": "error",
                "message": "Database not connected",
                "alerts": [],
                "alert_count": 0,
            }

        alerts = []

        try:
            with self.driver.session() as session:
                # 1. Create/Update the Case node
                session.run(
                    """
                    MERGE (c:Case {id: $cid})
                    SET c.title = $title,
                        c.last_updated = timestamp()
                    """,
                    cid=case_id,
                    title=case_title,
                )

                # 2. Process each entity and check for cross-case links
                for entity in entities:
                    e_type = entity["type"]
                    e_value = entity["value"]

                    # Skip garbage / empty values defensively
                    if not e_value or e_value in ("N/A", "Unknown"):
                        continue

                    query = """
                    MATCH (current_case:Case {id: $cid})
                    MERGE (e:Entity {value: $val, type: $type})
                    ON CREATE SET
                        e.first_seen = timestamp(),
                        e.case_count = 1
                    ON MATCH SET
                        e.case_count = coalesce(e.case_count, 0) + 1,
                        e.last_seen = timestamp()
                    MERGE (current_case)-[:HAS_ENTITY]->(e)

                    WITH e
                    MATCH (other_case:Case)-[:HAS_ENTITY]->(e)
                    WHERE other_case.id <> $cid
                    RETURN other_case.id   AS linked_case_id,
                           other_case.title AS linked_case_title
                    """

                    result = session.run(
                        query,
                        cid=case_id,
                        val=e_value,
                        type=e_type,
                    )

                    existing_links = [record.data() for record in result]

                    if existing_links:
                        alerts.append(
                            {
                                "suspicious_entity": e_value,
                                "entity_type": e_type,
                                "linked_to": existing_links,
                            }
                        )

            # 3. Success response
            return {
                "status": "success",
                "case_id": case_id,
                "alerts": alerts,
                "alert_count": len(alerts),
            }

        except Exception as e:
            # Ensure we always return a dict (never None)
            print(f"❌❌❌ NEO4J CRASH in ingest_case_data: {e}")
            return {
                "status": "error",
                "message": f"Neo4j query failed: {str(e)}",
                "case_id": case_id,
                "alerts": [],
                "alert_count": 0,
            }

    # 2. Get full network for a given Case

    def get_case_network(self, case_id: str):
        """
        Fetches all nodes and relationships connected to a given case_id
        for visualization.

        Returns:
          {
            "nodes": [
              { "id": "case-uuid", "label": "Case", "title": "Case File: ..." },
              { "id": "8289002887", "label": "PHONE", "title": "8289002887" },
              ...
            ],
            "links": [
              { "source": "case-uuid", "target": "8289002887" },
              { "source": "other-case-uuid", "target": "8289002887" },
              ...
            ]
          }
        """
        if not self.driver:
            return {"nodes": [], "links": []}

        query = """
        MATCH (c:Case {id: $cid})-[:HAS_ENTITY]->(e:Entity)
        OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(other:Case)
        RETURN c, e, other
        """

        nodes = {}
        links = []
        seen_links = set()  # to dedupe (source, target) pairs

        with self.driver.session() as session:
            results = session.run(query, cid=case_id)

            for record in results:
                case_node = record["c"]
                entity_node = record["e"]
                other_node = record["other"]

                # Case node (current)
                cid = case_node["id"]
                if cid not in nodes:
                    nodes[cid] = {
                        "id": cid,
                        "label": "Case",
                        "title": case_node.get("title", cid),
                    }

                # Entity node
                e_val = entity_node["value"]
                e_type = entity_node.get("type", "Entity")
                e_case_count = entity_node.get("case_count", 1)

                # Use raw value as id (for backward compatibility)
                if e_val not in nodes:
                    nodes[e_val] = {
                        "id": e_val,
                        "label": e_type,          # e.g., PHONE, EMAIL, BANK_ACCOUNT
                        "title": e_val,
                        "case_count": e_case_count,
                    }

                # Link current case -> entity
                link_key = (cid, e_val)
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links.append({"source": cid, "target": e_val})

                # Other Case node(s)
                if other_node:
                    other_cid = other_node["id"]
                    if other_cid not in nodes:
                        nodes[other_cid] = {
                            "id": other_cid,
                            "label": "Case",
                            "title": other_node.get("title", other_cid),
                        }

                    # Link other case -> same entity
                    other_link_key = (other_cid, e_val)
                    if other_link_key not in seen_links:
                        seen_links.add(other_link_key)
                        links.append(
                            {"source": other_cid, "target": e_val}
                        )

        return {"nodes": list(nodes.values()), "links": links}


# Create a single instance to be used by the app
graph_service = CaseFusionGraph()
