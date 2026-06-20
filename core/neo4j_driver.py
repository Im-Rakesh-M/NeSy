import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from config import URI, USERNAME, PASSWORD

class Neo4jConnection:
    """
    Singleton connection manager.
    
    Rationale: Opening a new driver for every operation
    wastes resources. One shared instance handles all 
    requests across all agents.
    """
    _instance = None

    def __init__(self):
        self.driver = GraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
            connection_timeout=30
        )
        print("[NEO4J] Driver initialized successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Neo4jConnection()
        return cls._instance

    def session(self):
        return self.driver.session()

    def close(self):
        self.driver.close()
        Neo4jConnection._instance = None
        print("[NEO4J] Connection closed.")