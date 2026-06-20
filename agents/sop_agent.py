import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any, List
from agents.base_agent import BaseAgent


class SOPAgent(BaseAgent):
    """
    Dynamic SOP compliance enforcer.

    Reads SOP rules FROM Neo4j at runtime — never hardcodes them.
    Evaluates every event payload against active SOP constraints.
    Issues SOP_VETO for critical violations.

    Key difference from RegulatoryAgent:
    - RegulatoryAgent enforces external law — hard veto, no override
    - SOPAgent enforces internal factory SOPs — can be escalated
      to human manager for exception approval

    Subscribes to all operational topics to catch SOP violations
    at the earliest possible point in the decision chain.
    """

    def __init__(self, message_bus):
        super().__init__("SOPAgent", message_bus)
        for topic in [
            "CASCADE_ALERT",
            "DELIVERY_RISK",
            "MACHINE_RISK",
            "BUFFER_ALERT",
            "RESTOCK_DIRECTIVE"
        ]:
            self.bus.subscribe(topic, self.perceive)

    async def perceive(self, payload: Dict[str, Any]):
        """Intercepts events and evaluates SOP compliance."""
        await self.decide(payload)

    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decision logic:
        1. Load all SOP rules from Neo4j graph
        2. Evaluate each rule against event context
        3. Apply priority ordering — highest priority rule wins
        4. Issue SOP_VETO for CRITICAL violations
        5. Issue WARNING for HIGH/MEDIUM violations
        6. Log every evaluation as AuditEntry
        """
        topic = context.get("_topic", "UNKNOWN")
        line_id = context.get("line_id")
        machine_id = context.get("machine_id")

        # Load SOP rules from graph — ordered by priority
        rules = self.query_graph("""
            MATCH (s:SOPRule)-[:TRIGGERS_ACTION]->(m:Mitigation)
            RETURN s.code AS code,
                   s.name AS name,
                   s.condition_field AS field,
                   s.condition_operator AS operator,
                   s.condition_value AS threshold,
                   s.severity AS severity,
                   s.priority AS priority,
                   m.vector AS action,
                   m.estimated_cost AS cost,
                   m.estimated_resolution_mins AS resolution_mins
            ORDER BY s.priority ASC
        """, {})

        if not rules:
            print(f"[{self.agent_name}] No SOP rules in graph.")
            return {"action": "NO_RULES"}

        print(f"[{self.agent_name}] Evaluating {len(rules)} "
              f"SOP rules against {topic} event")

        violations = []

        for rule in rules:
            breached = self._evaluate_rule(rule, context)
            if breached:
                violations.append(rule)
                # Stop at first violation — priority ordering means
                # highest priority rule takes precedence
                break

        if not violations:
            self.log_audit(
                event_type="SOP_EVALUATION",
                decision="ALL_SOP_COMPLIANT",
                confidence=1.0,
                rule_fired="ALL_RULES_CHECKED",
                status="COMPLIANT",
                cost=0.0
            )
            return {"action": "COMPLIANT", "topic": topic}

        # Violation found — take action based on severity
        violation = violations[0]
        rule_code = violation["code"]
        severity = violation["severity"]
        action = violation["action"]
        cost = violation["cost"]

        if severity == "CRITICAL":
            self.log_audit(
                event_type="SOP_VIOLATION",
                decision=f"SOP_VETO_{rule_code}",
                confidence=1.0,
                rule_fired=rule_code,
                status="SOP_VETO",
                cost=float(cost),
                alternative="Halt operations until resolved"
            )
            await self.publish("COMPLIANCE_BREACH", {
                "veto_type": "SOP_VETO",
                "rule_code": rule_code,
                "rule_name": violation["name"],
                "severity": severity,
                "line_id": line_id,
                "machine_id": machine_id,
                "prescribed_action": action,
                "estimated_cost": cost,
                "resolution_mins": violation["resolution_mins"],
                "overridable": True,
                "override_path": "HUMAN_MANAGER_APPROVAL"
            })
            print(f"[{self.agent_name}] SOP VETO — "
                  f"{rule_code} [{severity}] | "
                  f"Action: {action}")
            return {
                "action": "SOP_VETO",
                "rule": rule_code,
                "prescribed": action
            }

        # HIGH or MEDIUM — issue warning, don't veto
        self.log_audit(
            event_type="SOP_VIOLATION",
            decision=f"SOP_WARNING_{rule_code}",
            confidence=1.0,
            rule_fired=rule_code,
            status="WARNING",
            cost=float(cost),
            alternative="Monitor and schedule corrective action"
        )
        await self.publish("COMPLIANCE_BREACH", {
            "veto_type": "SOP_WARNING",
            "rule_code": rule_code,
            "rule_name": violation["name"],
            "severity": severity,
            "line_id": line_id,
            "prescribed_action": action,
            "estimated_cost": cost,
            "overridable": True
        })
        print(f"[{self.agent_name}] SOP WARNING — "
              f"{rule_code} [{severity}] | "
              f"Action: {action}")
        return {
            "action": "SOP_WARNING",
            "rule": rule_code
        }

    def _evaluate_rule(
        self, rule: Dict, context: Dict
    ) -> bool:
        """
        Evaluates a single SOP rule against event context.
        Returns True if rule condition is breached.

        Reads condition_field, condition_operator, condition_value
        from the rule node — never hardcoded.
        """
        field = rule.get("field")
        operator = rule.get("operator")
        threshold = rule.get("threshold")

        if not field or not operator or threshold is None:
            return False

        # Extract field value from context
        value = context.get(field)

        # Also check nested machine/line data if available
        if value is None:
            value = context.get(f"machine_{field}")
        if value is None:
            value = context.get(f"line_{field}")
        if value is None:
            return False

        try:
            value = float(value)
            threshold = float(threshold)
        except (TypeError, ValueError):
            return False

        if operator == "GREATER_THAN":
            return value > threshold
        elif operator == "LESS_THAN":
            return value < threshold
        elif operator == "EQUALS":
            return value == threshold
        elif operator == "GREATER_THAN_OR_EQUAL":
            return value >= threshold
        elif operator == "LESS_THAN_OR_EQUAL":
            return value <= threshold

        return False