from neo4j import GraphDatabase
from config import URI, USERNAME, PASSWORD

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD),
    connection_timeout=30
)

try:
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j Aura connected successfully' AS msg")
        print(result.single()["msg"])
except Exception as e:
    print(f"Connection failed: {e}")
finally:
    driver.close()