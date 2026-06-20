import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
from core.neo4j_driver import Neo4jConnection


class BaseAgent(ABC):
    """
    Abstract base class for all NESO-DT agents.

    Every agent must implement:
        perceive(payload) — process an incoming event
        decide(context)   — make a decision based on context

    Rationale for ABC pattern:
    Enforces a consistent contract across all agents.
    The orchestrator can call perceive() and decide() on
    any agent without knowing its concrete type — standard
    polymorphism for a maintainable multi-agent system.
    """

    def __init__(self, agent_name: str, message_bus):
        self.agent_name = agent_name
        self.bus = message_bus
        self.conn = Neo4jConnection.get_instance()
        self.decision_log = []

    @abstractmethod
    async def perceive(self, payload: Dict[str, Any]):
        """
        Process an incoming event from the message bus.
        Each agent implements its own perception logic.
        """
        pass

    @abstractmethod
    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision based on perceived context.
        Returns decision dict with action and rationale.
        """
        pass

    async def publish(self, topic: str, payload: Dict[str, Any]):
        """Helper — publish to message bus with sender tag."""
        payload['sender'] = self.agent_name
        await self.bus.publish(topic, payload)

    def log_audit(
        self,
        event_type: str,
        decision: str,
        confidence: float,
        rule_fired: str,
        status: str,
        cost: float = 0.0,
        alternative: str = "NONE"
    ):
        """
        Writes AuditEntry node to Neo4j.

        This is the neuro-symbolic fusion record —
        it captures both the neural confidence score
        AND the symbolic rule that fired, in one node.
        The EQA engine traverses these nodes to explain
        every decision in plain language.

        Parameters:
            event_type  : what triggered this decision
            decision    : what action was taken
            confidence  : neural model confidence (0-1)
            rule_fired  : symbolic SOP/regulatory rule code
            status      : COMPLIANT / VIOLATION / ESCALATED
            cost        : financial cost of action
            alternative : what alternative was considered
        """
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "agent_name": self.agent_name,
            "event_type": event_type,
            "decision": decision,
            "neural_confidence": round(float(confidence), 4),
            "symbolic_rule_fired": rule_fired,
            "compliance_status": status,
            "cost_of_action": round(float(cost), 2),
            "alternative_considered": alternative
        }

        # Write to Neo4j
        try:
            with self.conn.session() as session:
                session.run("""
                    CREATE (:AuditEntry {
                        id                  : $id,
                        timestamp           : $timestamp,
                        agent_name          : $agent_name,
                        event_type          : $event_type,
                        decision            : $decision,
                        neural_confidence   : $neural_confidence,
                        symbolic_rule_fired : $symbolic_rule_fired,
                        compliance_status   : $compliance_status,
                        cost_of_action      : $cost_of_action,
                        alternative_considered: $alternative_considered
                    })
                """, **entry)
        except Exception as e:
            print(f"[{self.agent_name}] Audit write failed: {e}")

        # Also keep in memory for fast EQA access
        self.decision_log.append(entry)
        print(f"[{self.agent_name}] "
              f"Decision logged: {decision} | "
              f"Rule: {rule_fired} | "
              f"Status: {status}")

    def get_decision_log(self) -> list:
        """Returns in-memory decision log for this agent."""
        return self.decision_log

    def query_graph(self, cypher: str, params: dict = None) -> list:
        """
        Helper for agents to query Neo4j directly.
        All graph reads go through here — single point
        for connection management.
        """
        with self.conn.session() as session:
            result = session.run(cypher, params or {})
            return [dict(r) for r in result]

    def update_belief(self, key: str, value: dict):
        """
        Persists agent belief to Neo4j.
        Beliefs survive across runs — this is what separates
        deliberative agents from reactive ones.
        """
        try:
            with self.conn.session() as session:
                session.run("""
                    MERGE (b:AgentBelief {
                        agent: $agent,
                        key: $key
                    })
                    SET b.value = $value,
                        b.updated = datetime(),
                        b.agent_name = $agent
                """, agent=self.agent_name,
                    key=key,
                    value=str(value))
        except Exception as e:
            print(f"[{self.agent_name}] Belief update failed: {e}")

    def get_belief(self, key: str) -> dict:
        """
        Retrieves agent belief from Neo4j.
        Returns empty dict if belief doesn't exist yet.
        """
        result = self.query_graph("""
            MATCH (b:AgentBelief {
                agent: $agent,
                key: $key
            })
            RETURN b.value AS value,
                b.updated AS updated
        """, {"agent": self.agent_name, "key": key})

        if not result:
            return {}

        try:
            return eval(result[0]["value"])
        except Exception:
            return {}    