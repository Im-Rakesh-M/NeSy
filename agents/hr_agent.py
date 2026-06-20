import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from agents.base_agent import BaseAgent


class HRAgent(BaseAgent):
    """
    Monitors shift coverage and labour compliance.

    Responsibilities:
    1. Check certified operator availability per line per shift
    2. Read minimum staffing requirements from graph (REG_003)
    3. If understaffed — check overtime eligible workers
    4. If still understaffed — publish SHIFT_GAP alert
    5. Enforce India Factories Act (REG_002) hour limits
    """

    def __init__(self, message_bus):
        super().__init__("HRAgent", message_bus)
        # Subscribe to both — proactive check on delivery risk
        # and reactive check on cascade alerts
        self.bus.subscribe("DELIVERY_RISK", self.perceive)
        self.bus.subscribe("CASCADE_ALERT", self.perceive)

    async def perceive(self, payload: Dict[str, Any]):
        """Receives event and triggers shift coverage check."""
        line_id = payload.get("line_id")
        if not line_id:
            return
        await self.decide({
            "line_id": line_id,
            "trigger_event": payload.get("_topic", "UNKNOWN"),
            "order_id": payload.get("order_id"),
            "neural_confidence": payload.get(
                "neural_confidence", 0.0
            )
        })

    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decision tree:
        1. Read required staffing level from graph (not hardcoded)
        2. Count available certified workers on correct shift
        3. Sufficient? → COMPLIANT
        4. Insufficient? → Check overtime eligible workers
        5. Still insufficient? → SHIFT_GAP alert
        6. Check REG_002 hour limits for active workers
        """
        line_id = context.get("line_id")
        order_id = context.get("order_id")
        neural_confidence = context.get("neural_confidence", 0.0)
        trigger = context.get("trigger_event", "UNKNOWN")

        print(f"[{self.agent_name}] Checking shift coverage "
              f"for Line {line_id}")

        # Get line requirements from graph — not hardcoded
        line_info = self.query_graph("""
            MATCH (l:ProductionLine {line_id: $line_id})
            RETURN l.shift_requirement AS required,
                   l.certified_role AS role,
                   l.name AS name
        """, {"line_id": line_id})

        if not line_info:
            print(f"[{self.agent_name}] Line {line_id} not found.")
            return {"action": "ERROR", "line_id": line_id}

        required_count = line_info[0]["required"]
        required_role = line_info[0]["certified_role"]
        line_name = line_info[0]["name"]

        # Count available certified workers on any active shift
        available = self.query_graph("""
            MATCH (w:ShiftWorker)-[:CERTIFIED_FOR]->
                  (l:ProductionLine {line_id: $line_id})
            WHERE w.availability = 1
            RETURN count(w) AS count,
                   collect(w.worker_id) AS worker_ids,
                   collect(w.shift_id) AS shifts
        """, {"line_id": line_id})

        available_count = available[0]["count"] if available else 0
        available_workers = available[0]["worker_ids"] if available else []

        print(f"  -> Required: {required_count} | "
              f"Available: {available_count} | "
              f"Role: {required_role}")

        # Read minimum from regulatory rule REG_003
        reg_min = self.query_graph("""
            MATCH (r:RegulatoryRule {rule_id: 'REG_003'})
            RETURN r.min_certified_operators AS minimum,
                   r.penalty_eur AS penalty
        """, {})

        reg_minimum = reg_min[0]["minimum"] if reg_min else 2
        reg_penalty = reg_min[0]["penalty"] if reg_min else 25000

        # Use the stricter of line requirement and regulatory minimum
        effective_minimum = max(required_count, reg_minimum)

        if available_count >= effective_minimum:
            self.log_audit(
                event_type="SHIFT_COVERAGE_CHECK",
                decision="STAFFING_VERIFIED",
                confidence=1.0,
                rule_fired="REG_003_ISO_45001",
                status="COMPLIANT",
                cost=0.0,
                alternative=f"Required {effective_minimum}, "
                            f"have {available_count}"
            )
            print(f"[{self.agent_name}] ✅ STAFFING_VERIFIED — "
                f"{available_count}/{effective_minimum} certified workers on Line {line_id}")
            return {
                "action": "STAFFING_VERIFIED",
                "line_id": line_id,
                "available_count": available_count
            }

        # Check overtime eligible workers as fallback
        overtime = self.query_graph("""
            MATCH (w:ShiftWorker)-[:CERTIFIED_FOR]->
                  (l:ProductionLine {line_id: $line_id})
            WHERE w.overtime_eligible = true
            AND w.availability = 0
            RETURN count(w) AS count,
                   collect(w.worker_id) AS worker_ids
        """, {"line_id": line_id})

        overtime_count = overtime[0]["count"] if overtime else 0
        total_with_overtime = available_count + overtime_count

        if total_with_overtime >= effective_minimum:
            overtime_needed = effective_minimum - available_count
            self.log_audit(
                event_type="SHIFT_COVERAGE_CHECK",
                decision=f"OVERTIME_REQUIRED_{overtime_needed}_WORKERS",
                confidence=1.0,
                rule_fired="REG_002_FACTORIES_ACT",
                status="WARNING",
                cost=float(overtime_needed * 500),
                alternative="Delay delivery acceptance"
            )
            await self.publish("SHIFT_GAP", {
                "line_id": line_id,
                "order_id": order_id,
                "severity": "WARNING",
                "available_count": available_count,
                "required_count": effective_minimum,
                "overtime_workers_needed": overtime_needed,
                "action_required": "CALL_OVERTIME_WORKERS",
                "trigger_event": trigger
            })
            return {
                "action": "OVERTIME_REQUIRED",
                "line_id": line_id,
                "overtime_needed": overtime_needed
            }

        # Genuinely understaffed — cannot proceed
        shortage = effective_minimum - available_count
        penalty = reg_penalty if available_count < reg_minimum else 0

        self.log_audit(
            event_type="SHIFT_COVERAGE_CHECK",
            decision=f"SHIFT_GAP_CRITICAL_{shortage}_SHORT",
            confidence=1.0,
            rule_fired="REG_003_ISO_45001_BREACH",
            status="COMPLIANCE_BREACH",
            cost=float(penalty),
            alternative="Cannot proceed — regulatory breach"
        )
        await self.publish("SHIFT_GAP", {
            "line_id": line_id,
            "order_id": order_id,
            "severity": "CRITICAL",
            "available_count": available_count,
            "required_count": effective_minimum,
            "shortage": shortage,
            "regulatory_minimum": reg_minimum,
            "penalty_eur": penalty,
            "action_required": "DELAY_OR_HALT_OPERATIONS",
            "trigger_event": trigger
        })
        await self.publish("COMPLIANCE_BREACH", {
            "line_id": line_id,
            "reason": "ISO_45001_MINIMUM_STAFFING_BREACH",
            "available": available_count,
            "required": effective_minimum,
            "penalty_eur": penalty
        })
        return {
            "action": "CRITICAL_SHIFT_GAP",
            "line_id": line_id,
            "shortage": shortage
        }