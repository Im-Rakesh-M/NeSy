import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from agents.base_agent import BaseAgent


class InventoryAgent(BaseAgent):
    """
    Monitors buffer levels and manages restocking directives.

    Responsibilities:
    1. Receive BUFFER_ALERT from Logistics Agent
    2. Assess buffer status on target production line
    3. If reroute action received — confirm backup supplier
       exists and issue RESTOCK_DIRECTIVE
    4. Update buffer levels in Neo4j after restock
    5. Monitor buffer pct and alert if approaching
       safety stock threshold
    """

    BUFFER_WARNING_PCT = 30   # Warning threshold
    BUFFER_CRITICAL_PCT = 15  # Critical threshold

    def __init__(self, message_bus):
        super().__init__("InventoryAgent", message_bus)
        self.bus.subscribe("BUFFER_ALERT", self.perceive)
        self.bus.subscribe("RESTOCK_DIRECTIVE", self.handle_restock)

    async def perceive(self, payload: Dict[str, Any]):
        """Receives buffer alert and triggers decision."""
        await self.decide(payload)

    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decision tree:
        1. What is current buffer status on line?
        2. Is action REROUTE_TO_BACKUP?
           YES → verify backup supplier, issue restock directive
        3. Is buffer below critical threshold?
           YES → escalate CASCADE_ALERT
        4. Is buffer below warning threshold?
           YES → publish warning, monitor
        """
        line_id = context.get("line_id")
        order_id = context.get("order_id")
        severity = context.get("severity", "WARNING")
        action = context.get("action", "MONITOR")
        backup_supplier_id = context.get("backup_supplier_id")
        neural_confidence = context.get("neural_confidence", 0.0)
        part_type = context.get("part_type")

        print(f"[{self.agent_name}] Buffer assessment for "
              f"Line {line_id} | Action={action}")

        # Get current buffer state from graph
        buffer_state = self.query_graph("""
            MATCH (l:ProductionLine {line_id: $line_id})
            RETURN l.buffer_units AS buffer_units,
                   l.buffer_capacity AS capacity,
                   l.safety_stock_units AS safety_stock,
                   l.buffer_pct AS buffer_pct,
                   l.part_type AS part_type,
                   l.cycle_time_mins AS cycle_time
        """, {"line_id": line_id})

        if not buffer_state:
            print(f"[{self.agent_name}] Line {line_id} not in graph.")
            return {"action": "ERROR", "line_id": line_id}

        buf = buffer_state[0]
        buffer_pct = buf["buffer_pct"]
        buffer_units = buf["buffer_units"]
        safety_stock = buf["safety_stock"]
        line_part_type = buf["part_type"]

        # Handle reroute action from Logistics Agent
        if action == "REROUTE_TO_BACKUP" and backup_supplier_id:
            # Verify backup supplier in graph
            supplier = self.query_graph("""
                MATCH (s:Supplier {supplier_id: $sid})
                      -[:BACKUP_SUPPLIER_FOR]->(l:ProductionLine {line_id: $line_id})
                RETURN s.supplier_id AS supplier_id,
                       s.name AS name,
                       s.reliability_score AS reliability,
                       s.lead_time_days AS lead_time,
                       s.cost_per_unit AS cost
            """, {"sid": backup_supplier_id, "line_id": line_id})

            if not supplier:
                self.log_audit(
                    event_type="BUFFER_ALERT",
                    decision="BACKUP_SUPPLIER_NOT_VERIFIED",
                    confidence=neural_confidence,
                    rule_fired="SUPPLIER_VERIFICATION_CHECK",
                    status="ESCALATED",
                    cost=0.0,
                    alternative="No verified backup found"
                )
                await self.publish("CASCADE_ALERT", {
                    "line_id": line_id,
                    "order_id": order_id,
                    "severity": "CRITICAL",
                    "reason": "BACKUP_SUPPLIER_UNVERIFIED",
                    "action_required": "HUMAN_INTERVENTION"
                })
                return {"action": "ESCALATED", "line_id": line_id}

            sup = supplier[0]

            self.log_audit(
                event_type="BUFFER_ALERT",
                decision=f"RESTOCK_DIRECTIVE_ISSUED_{sup['supplier_id']}",
                confidence=neural_confidence,
                rule_fired="BACKUP_SUPPLIER_PROTOCOL",
                status="COMPLIANT",
                cost=float(sup["cost"]),
                alternative=f"Primary supplier delayed"
            )

            # Issue restock directive to line
            await self.publish("RESTOCK_DIRECTIVE", {
                "line_id": line_id,
                "order_id": order_id,
                "supplier_id": sup["supplier_id"],
                "supplier_name": sup["name"],
                "part_type": line_part_type,
                "units_required": int(safety_stock * 1.5),
                "cost_per_unit": sup["cost"],
                "neural_confidence": neural_confidence,
                "priority": "HIGH"
            })
            return {
                "action": "RESTOCK_DIRECTIVE_ISSUED",
                "supplier": sup["supplier_id"],
                "line_id": line_id
            }

        # Monitor buffer thresholds
        if buffer_pct <= self.BUFFER_CRITICAL_PCT:
            self.log_audit(
                event_type="BUFFER_ALERT",
                decision="CRITICAL_BUFFER_ALERT",
                confidence=neural_confidence,
                rule_fired="SOP_BUFFER_CRITICAL",
                status="VIOLATION_DETECTED",
                cost=0.0,
                alternative="Wait for scheduled delivery"
            )
            await self.publish("CASCADE_ALERT", {
                "line_id": line_id,
                "severity": "CRITICAL",
                "reason": "BUFFER_BELOW_CRITICAL_THRESHOLD",
                "buffer_pct": buffer_pct,
                "buffer_units": buffer_units,
                "safety_stock": safety_stock,
                "action_required": "EMERGENCY_RESTOCK"
            })
            return {
                "action": "CRITICAL_BUFFER_ALERT",
                "line_id": line_id
            }

        if buffer_pct <= self.BUFFER_WARNING_PCT:
            self.log_audit(
                event_type="BUFFER_ALERT",
                decision="BUFFER_WARNING_ISSUED",
                confidence=neural_confidence,
                rule_fired="BUFFER_WARNING_THRESHOLD",
                status="WARNING",
                cost=0.0,
                alternative="Continue monitoring"
            )
            return {
                "action": "BUFFER_WARNING",
                "line_id": line_id,
                "buffer_pct": buffer_pct
            }

        # Buffer healthy
        self.log_audit(
            event_type="BUFFER_ALERT",
            decision="BUFFER_HEALTHY",
            confidence=neural_confidence,
            rule_fired="NONE",
            status="COMPLIANT",
            cost=0.0
        )
        return {"action": "BUFFER_HEALTHY", "line_id": line_id}

    async def handle_restock(self, payload: Dict[str, Any]):
        """
        Handles restock completion — updates buffer
        level in Neo4j after parts arrive on line.
        """
        line_id = payload.get("line_id")
        units_delivered = payload.get("units_delivered", 0)

        if not line_id or not units_delivered:
            return

        # Update buffer in graph
        with self.conn.session() as session:
            session.run("""
                MATCH (l:ProductionLine {line_id: $line_id})
                SET l.buffer_units = l.buffer_units + $units,
                    l.buffer_pct = toFloat(
                        l.buffer_units + $units
                    ) / toFloat(l.buffer_capacity) * 100
            """, {"line_id": line_id, "units": units_delivered})

        print(f"[{self.agent_name}] Buffer updated for "
              f"Line {line_id} +{units_delivered} units")

        self.log_audit(
            event_type="RESTOCK_COMPLETE",
            decision=f"BUFFER_UPDATED_{line_id}",
            confidence=1.0,
            rule_fired="RESTOCK_CONFIRMATION",
            status="COMPLIANT",
            cost=0.0
        )