import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from agents.base_agent import BaseAgent
from datetime import datetime


class MaintenanceAgent(BaseAgent):
    """
    Monitors machine health signals and enforces
    maintenance SOPs and regulatory compliance.

    Responsibilities:
    1. Receive machine risk predictions from neural layer
    2. Query graph for which SOP rule applies
    3. Check regulatory maintenance window (REG_001)
    4. Decide: veto delivery acceptance / schedule maintenance
       / allow operation
    5. Notify line agent and regulatory agent via message bus
    """

    # Risk thresholds for agent decisions
    CRITICAL_FAILURE_PROB = 0.80
    HIGH_FAILURE_PROB = 0.50
    REGULATORY_HOUR_LIMIT = 90  # EU Machinery Directive

    def __init__(self, message_bus):
        super().__init__("MaintenanceAgent", message_bus)
        self.bus.subscribe("MACHINE_RISK", self.perceive)

    async def perceive(self, payload: Dict[str, Any]):
        """Receives machine risk event and triggers decision."""
        await self.decide(payload)

    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decision tree:
        1. Is failure probability CRITICAL (>0.80)?
           YES → immediate veto, halt line
        2. Is failure probability HIGH (>0.50)?
           YES → schedule maintenance, hold next delivery
        3. Is regulatory hour limit approaching?
           YES → flag mandatory maintenance window
        4. What SOP rule applies? Read from graph.
        5. Log audit entry with neural + symbolic context
        """
        machine_id = context.get("machine_id")
        prob_fail = context.get("failure_probability", 0.0)
        risk_level = context.get("risk_level", "LOW")
        prediction_set = context.get("prediction_set", [])
        confidence = context.get("confidence", "LOW")
        wear_criticality = context.get("wear_criticality", 0.0)
        thermal_stress = context.get("thermal_stress", 0.0)
        operation_hours = context.get("operation_hours", 0.0)

        print(f"[{self.agent_name}] Assessing Machine {machine_id} | "
              f"P(fail)={prob_fail:.2f} | "
              f"Risk={risk_level}")

        # Get line this machine operates on
        line_info = self.query_graph("""
            MATCH (m:ProductionHour {uid: $uid})
                  -[:OPERATES_ON]->(l:ProductionLine)
            RETURN l.line_id AS line_id,
                   l.name AS line_name,
                   l.part_type AS part_type
        """, {"uid": machine_id})

        line_id = line_info[0]["line_id"] if line_info else "UNKNOWN"
        line_name = line_info[0]["line_name"] if line_info else "UNKNOWN"

        # Get applicable SOP rule from graph — not hardcoded
        sop_rules = self.query_graph("""
            MATCH (m:ProductionHour {uid: $uid})
                  -[:VIOLATED_CONSTRAINT]->(s:SOPRule)
                  -[:TRIGGERS_ACTION]->(mit:Mitigation)
            RETURN s.code AS rule_code,
                   s.severity AS severity,
                   mit.vector AS action,
                   mit.estimated_cost AS cost,
                   mit.estimated_resolution_mins AS resolution_mins
            ORDER BY s.priority ASC
            LIMIT 1
        """, {"uid": machine_id})

        rule_code = sop_rules[0]["rule_code"] if sop_rules else "NO_SOP_VIOLATION"
        mitigation = sop_rules[0]["action"] if sop_rules else "CONTINUE_MONITORING"
        mit_cost = sop_rules[0]["cost"] if sop_rules else 0.0
        resolution_mins = sop_rules[0]["resolution_mins"] if sop_rules else 0

        # Check regulatory hour limit (REG_001)
        regulatory_breach = operation_hours >= self.REGULATORY_HOUR_LIMIT
        approaching_limit = operation_hours >= (self.REGULATORY_HOUR_LIMIT * 0.90)

        # Decision path 1 — CRITICAL failure probability
        if prob_fail >= self.CRITICAL_FAILURE_PROB:
            self.log_audit(
                event_type="MACHINE_RISK",
                decision=f"HALT_LINE_{line_id}",
                confidence=prob_fail,
                rule_fired=rule_code,
                status="VETO_RAISED",
                cost=float(mit_cost),
                alternative="Continue at risk of failure"
            )
            await self.publish("CASCADE_ALERT", {
                "machine_id": machine_id,
                "line_id": line_id,
                "line_name": line_name,
                "severity": "CRITICAL",
                "reason": "HIGH_FAILURE_PROBABILITY",
                "prob_fail": prob_fail,
                "rule_fired": rule_code,
                "mitigation": mitigation,
                "resolution_mins": resolution_mins,
                "action_required": "HALT_LINE_AND_SCHEDULE_MAINTENANCE"
            })
            return {
                "action": "HALT_LINE",
                "machine_id": machine_id,
                "line_id": line_id
            }

        # Decision path 2 — regulatory breach
        if regulatory_breach:
            self.log_audit(
                event_type="REGULATORY_CHECK",
                decision="MANDATORY_MAINTENANCE_REQUIRED",
                confidence=prob_fail,
                rule_fired="REG_001_EU_MACHINERY_DIRECTIVE",
                status="COMPLIANCE_BREACH",
                cost=50000.0,
                alternative="Document exception — not permitted"
            )
            await self.publish("COMPLIANCE_BREACH", {
                "machine_id": machine_id,
                "line_id": line_id,
                "reason": "REG_001_HOUR_LIMIT_EXCEEDED",
                "operation_hours": operation_hours,
                "limit": self.REGULATORY_HOUR_LIMIT,
                "action_required": "IMMEDIATE_MAINTENANCE",
                "penalty_eur": 50000
            })
            return {
                "action": "MANDATORY_MAINTENANCE",
                "machine_id": machine_id
            }

        # Decision path 3 — HIGH failure probability
        if prob_fail >= self.HIGH_FAILURE_PROB:
            self.log_audit(
                event_type="MACHINE_RISK",
                decision=f"SCHEDULE_MAINTENANCE_{line_id}",
                confidence=prob_fail,
                rule_fired=rule_code,
                status="MAINTENANCE_SCHEDULED",
                cost=float(mit_cost),
                alternative="Continue monitoring"
            )
            await self.publish("CASCADE_ALERT", {
                "machine_id": machine_id,
                "line_id": line_id,
                "severity": "HIGH",
                "reason": "ELEVATED_FAILURE_PROBABILITY",
                "prob_fail": prob_fail,
                "rule_fired": rule_code,
                "mitigation": mitigation,
                "action_required": "SCHEDULE_NEXT_SHIFT_MAINTENANCE"
            })
            return {
                "action": "SCHEDULE_MAINTENANCE",
                "machine_id": machine_id
            }

        # Decision path 4 — approaching regulatory limit
        if approaching_limit:
            self.log_audit(
                event_type="REGULATORY_CHECK",
                decision="FLAG_MAINTENANCE_WINDOW",
                confidence=prob_fail,
                rule_fired="REG_001_APPROACHING_LIMIT",
                status="WARNING",
                cost=0.0,
                alternative="Continue and monitor"
            )
            await self.publish("COMPLIANCE_BREACH", {
                "machine_id": machine_id,
                "line_id": line_id,
                "reason": "REG_001_APPROACHING_LIMIT",
                "operation_hours": operation_hours,
                "hours_remaining": self.REGULATORY_HOUR_LIMIT - operation_hours,
                "action_required": "PLAN_MAINTENANCE_WINDOW"
            })
            return {
                "action": "FLAG_MAINTENANCE_WINDOW",
                "machine_id": machine_id
            }

        # Decision path 5 — low risk, monitor only
        self.log_audit(
            event_type="MACHINE_RISK",
            decision="CONTINUE_OPERATION",
            confidence=prob_fail,
            rule_fired="NONE",
            status="COMPLIANT",
            cost=0.0,
            alternative="No action needed"
        )

        # Update machine health belief
        machine_belief_key = f"machine_{machine_id}_health"
        current = self.get_belief(machine_belief_key)

        self.update_belief(machine_belief_key, {
            "machine_id": machine_id,
            "line_id": line_id,
            "last_prob_fail": prob_fail,
            "last_risk_level": risk_level,
            "check_count": current.get("check_count", 0) + 1,
            "last_checked": datetime.utcnow().isoformat(),
            "consecutive_low_risk": (
                current.get("consecutive_low_risk", 0) + 1
                if risk_level == "LOW"
                else 0
            ),
            "consecutive_high_risk": (
                current.get("consecutive_high_risk", 0) + 1
                if risk_level in ["HIGH", "CRITICAL"]
                else 0
            )
        })
        print(f"[{self.agent_name}] Belief updated: "
            f"machine {machine_id} check #{current.get('check_count', 0) + 1}")


        return {
            "action": "CONTINUE_OPERATION",
            "machine_id": machine_id
        }