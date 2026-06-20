import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any, List
from agents.base_agent import BaseAgent


class RegulatoryAgent(BaseAgent):
    """
    Hard-veto regulatory compliance enforcer.

    This agent is the only one that can issue a HARD VETO —
    a decision that cannot be overridden by any other agent.

    Rationale for hard veto:
    Regulatory violations carry criminal and financial penalties.
    Unlike SOP rules which can have exceptions, regulatory rules
    are absolute. EU Machinery Directive, India Factories Act,
    and ISO standards have no override path in production.

    Subscribes to ALL topics — intercepts any event that
    could trigger a regulatory breach before it executes.
    """

    def __init__(self, message_bus):
        super().__init__("RegulatoryAgent", message_bus)
        # Subscribe to all topics — regulatory check on everything
        for topic in [
            "COMPLIANCE_BREACH",
            "CASCADE_ALERT",
            "DELIVERY_RISK",
            "MACHINE_RISK",
            "BUFFER_ALERT",
            "SHIFT_GAP",
            "RESTOCK_DIRECTIVE"
        ]:
            self.bus.subscribe(topic, self.perceive)

        # Cache regulatory rules from graph at startup
        self._rules_cache: List[Dict] = []

    async def perceive(self, payload: Dict[str, Any]):
        """Intercepts every event and checks for regulatory risk."""
        topic = payload.get("_topic", "UNKNOWN")

        # Only act on topics that have regulatory implications
        if topic in ["COMPLIANCE_BREACH", "SHIFT_GAP", "CASCADE_ALERT"]:
            await self.decide(payload)

    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Regulatory decision logic:
        1. Load applicable rules from Neo4j
        2. Check each rule against the event context
        3. If hard constraint violated — issue HARD VETO
        4. If soft constraint violated — issue WARNING
        5. Log every check as AuditEntry
        """
        reason = context.get("reason", "UNKNOWN")
        line_id = context.get("line_id")
        machine_id = context.get("machine_id")
        topic = context.get("_topic", "UNKNOWN")

        print(f"[{self.agent_name}] Regulatory check | "
              f"Topic={topic} | Reason={reason}")

        # Load all regulatory rules from graph
        rules = self.query_graph("""
            MATCH (r:RegulatoryRule)
            RETURN r.rule_id AS rule_id,
                   r.name AS name,
                   r.description AS description,
                   r.jurisdiction AS jurisdiction,
                   r.penalty_eur AS penalty,
                   r.hard_constraint AS hard_constraint,
                   r.applies_to AS applies_to
        """, {})

        if not rules:
            print(f"[{self.agent_name}] No regulatory rules "
                  f"found in graph.")
            return {"action": "NO_RULES_FOUND"}

        vetoes = []
        warnings = []

        for rule in rules:
            breach = self._evaluate_rule(rule, context)
            if breach:
                if rule["hard_constraint"]:
                    vetoes.append(rule)
                else:
                    warnings.append(rule)

        # Issue hard vetoes first
        if vetoes:
            veto_rules = [r["rule_id"] for r in vetoes]
            total_penalty = sum(r["penalty"] for r in vetoes)

            self.log_audit(
                event_type="REGULATORY_VETO",
                decision=f"HARD_VETO_ISSUED_{'+'.join(veto_rules)}",
                confidence=1.0,
                rule_fired="+".join(veto_rules),
                status="HARD_VETO",
                cost=float(total_penalty),
                alternative="No override permitted by law"
            )

            await self.publish("COMPLIANCE_BREACH", {
                "veto_type": "HARD_VETO",
                "violated_rules": veto_rules,
                "line_id": line_id,
                "machine_id": machine_id,
                "total_penalty_eur": total_penalty,
                "action_required": "IMMEDIATE_HALT",
                "overridable": False,
                "jurisdictions": [r["jurisdiction"] for r in vetoes],
                "descriptions": [r["description"] for r in vetoes]
            })

            print(f"[{self.agent_name}] "
                  f"HARD VETO issued for rules: {veto_rules} | "
                  f"Total penalty: €{total_penalty:,}")

            return {
                "action": "HARD_VETO",
                "rules": veto_rules,
                "penalty": total_penalty
            }

        # Issue soft warnings
        if warnings:
            warn_rules = [r["rule_id"] for r in warnings]

            self.log_audit(
                event_type="REGULATORY_WARNING",
                decision=f"WARNING_ISSUED_{'+'.join(warn_rules)}",
                confidence=1.0,
                rule_fired="+".join(warn_rules),
                status="WARNING",
                cost=0.0,
                alternative="Corrective action recommended"
            )

            print(f"[{self.agent_name}] "
                  f"Regulatory WARNING for rules: {warn_rules}")

            return {
                "action": "WARNING",
                "rules": warn_rules
            }

        # All clear
        self.log_audit(
            event_type="REGULATORY_CHECK",
            decision="ALL_CLEAR",
            confidence=1.0,
            rule_fired="ALL_RULES_CHECKED",
            status="COMPLIANT",
            cost=0.0
        )
        return {"action": "COMPLIANT"}

    def _evaluate_rule(
        self, rule: Dict, context: Dict
    ) -> bool:
        """
        Evaluates a single regulatory rule against event context.
        Returns True if rule is breached.

        Maps rule_id to specific context fields:
        REG_001 — machine operation hours
        REG_002 — worker consecutive hours
        REG_003 — certified operator count
        REG_004 — hazmat certification
        REG_005 — quality inspection presence
        """
        rule_id = rule["rule_id"]
        reason = context.get("reason", "")

        if rule_id == "REG_001":
            # EU Machinery Directive — operation hours
            return "REG_001" in reason or \
                   "HOUR_LIMIT" in reason

        elif rule_id == "REG_002":
            # India Factories Act — worker hours
            return "REG_002" in reason or \
                   "CONSECUTIVE_HOURS" in reason

        elif rule_id == "REG_003":
            # ISO 45001 — minimum certified operators
            return "ISO_45001" in reason or \
                   "STAFFING_BREACH" in reason or \
                   context.get("available", 99) < \
                   context.get("required", 0)

        elif rule_id == "REG_004":
            # REACH — hazmat certification
            return "HAZMAT" in reason or \
                   "REACH" in reason

        elif rule_id == "REG_005":
            # ISO 9001 — quality inspection
            return "QUALITY" in reason or \
                   "ISO_9001" in reason

        return False