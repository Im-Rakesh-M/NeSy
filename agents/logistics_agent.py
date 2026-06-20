import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from agents.base_agent import BaseAgent
from datetime import datetime


class LogisticsAgent(BaseAgent):
    """
    Monitors delivery risk signals and orchestrates
    supplier rerouting when JIT delivery is threatened.

    Responsibilities:
    1. Receive delivery risk predictions from neural layer
    2. Check buffer levels on receiving production line
    3. If delay > threshold AND buffer < safety stock:
       - Find alternate supplier from graph
       - Check HR coverage on target line
       - Trigger rerouting or escalate to CASCADE_ALERT
    4. Log every decision as AuditEntry in Neo4j
    """

    DELAY_THRESHOLD_MINS = 10  # JIT threshold from dissertation spec

    def __init__(self, message_bus):
        super().__init__("LogisticsAgent", message_bus)
        self.bus.subscribe("DELIVERY_RISK", self.perceive)

    async def perceive(self, payload: Dict[str, Any]):
        """Receives delivery risk event and triggers decision."""
        await self.decide(payload)

    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core decision logic for delivery risk management.

        Decision tree:
        1. Is delay predicted with HIGH confidence?
        2. Is delay > 10 min JIT threshold?
        3. Is buffer below safety stock on target line?
           YES → find alternate supplier from graph
           NO  → monitor only
        4. Is alternate supplier available?
           YES → trigger reroute
           NO  → escalate CASCADE_ALERT
        """
        order_id = context.get("order_id")
        line_id = context.get("line_id")
        prob_late = context.get("late_probability", 0.0)
        delay_mins = context.get("delay_mins", 0.0)
        prediction_set = context.get("prediction_set", [])
        confidence = context.get("confidence", "LOW")
        part_type = context.get("part_type")

        print(f"[{self.agent_name}] Assessing Order {order_id} "
              f"→ Line {line_id} | "
              f"P(late)={prob_late:.2f} | "
              f"Delay={delay_mins:.0f}mins")

        # Gate 1 — is delay predicted with meaningful confidence?
        if 1 not in prediction_set or prob_late < 0.5:
            self.log_audit(
                event_type="DELIVERY_RISK",
                decision="MONITOR_ONLY",
                confidence=prob_late,
                rule_fired="NONE",
                status="COMPLIANT",
                cost=0.0,
                alternative="No delay predicted"
            )
            return {"action": "MONITOR_ONLY", "order_id": order_id}

        # Gate 2 — is delay beyond JIT threshold?
        if delay_mins <= self.DELAY_THRESHOLD_MINS:
            self.log_audit(
                event_type="DELIVERY_RISK",
                decision="BUFFER_CAN_ABSORB",
                confidence=prob_late,
                rule_fired="JIT_10MIN_THRESHOLD",
                status="COMPLIANT",
                cost=0.0,
                alternative="Delay within buffer window"
            )
            return {"action": "BUFFER_CAN_ABSORB", "order_id": order_id}

        # Gate 3 — check buffer level on target line
        buffer_status = self.query_graph("""
            MATCH (l:ProductionLine {line_id: $line_id})
            RETURN l.buffer_units AS buffer,
                   l.safety_stock_units AS safety_stock,
                   l.buffer_pct AS buffer_pct,
                   l.cycle_time_mins AS cycle_time
        """, {"line_id": line_id})

        if not buffer_status:
            print(f"[{self.agent_name}] Line {line_id} not found in graph.")
            return {"action": "ERROR", "order_id": order_id}

        buffer = buffer_status[0]
        buffer_units = buffer["buffer"]
        safety_stock = buffer["safety_stock"]
        cycle_time = buffer["cycle_time"]

        # Calculate buffer coverage window
        buffer_coverage_mins = (
            (buffer_units - safety_stock) * cycle_time
        )

        if buffer_coverage_mins >= delay_mins:
            # Buffer can absorb the delay
            self.log_audit(
                event_type="DELIVERY_RISK",
                decision="BUFFER_ABSORBS_DELAY",
                confidence=prob_late,
                rule_fired="BUFFER_COVERAGE_CHECK",
                status="COMPLIANT",
                cost=0.0,
                alternative=f"Buffer covers {buffer_coverage_mins:.0f} mins"
            )
            # Notify inventory agent to monitor
            await self.publish("BUFFER_ALERT", {
                "line_id": line_id,
                "order_id": order_id,
                "severity": "WARNING",
                "buffer_units": buffer_units,
                "safety_stock": safety_stock,
                "delay_mins": delay_mins,
                "neural_confidence": prob_late
            })
            return {"action": "BUFFER_ABSORBS_DELAY",
                    "order_id": order_id}

        # Gate 4 — buffer insufficient, find alternate supplier
        alt_suppliers = self.query_graph("""
            MATCH (s:Supplier)-[:BACKUP_SUPPLIER_FOR]->(l:ProductionLine {line_id: $line_id})
            WHERE s.status = 'ACTIVE'
            RETURN s.supplier_id AS supplier_id,
                   s.name AS name,
                   s.reliability_score AS reliability,
                   s.lead_time_days AS lead_time,
                   s.cost_per_unit AS cost
            ORDER BY s.reliability_score DESC
        """, {"line_id": line_id})

        if not alt_suppliers:
            # No backup supplier — escalate
            self.log_audit(
                event_type="DELIVERY_RISK",
                decision="ESCALATE_NO_BACKUP_SUPPLIER",
                confidence=prob_late,
                rule_fired="SUPPLIER_AVAILABILITY_CHECK",
                status="ESCALATED",
                cost=0.0,
                alternative="No backup supplier available"
            )
            await self.publish("CASCADE_ALERT", {
                "line_id": line_id,
                "order_id": order_id,
                "severity": "CRITICAL",
                "reason": "NO_BACKUP_SUPPLIER",
                "delay_mins": delay_mins,
                "neural_confidence": prob_late,
                "sender": self.agent_name
            })
            return {"action": "CASCADE_ALERT", "order_id": order_id}

        # Best backup supplier found
        best_supplier = alt_suppliers[0]

        self.log_audit(
            event_type="DELIVERY_RISK",
            decision=f"REROUTE_TO_{best_supplier['supplier_id']}",
            confidence=prob_late,
            rule_fired="SUPPLIER_REROUTING_PROTOCOL",
            status="COMPLIANT",
            cost=float(best_supplier['cost']),
            alternative=f"Primary supplier delayed {delay_mins:.0f}mins"
        )

        # Update supplier reliability belief
        supplier_id = best_supplier["supplier_id"]
        belief_key = f"{supplier_id}_performance"
        current = self.get_belief(belief_key)

        self.update_belief(belief_key, {
            "supplier_id": supplier_id,
            "supplier_name": best_supplier["name"],
            "reroute_count": current.get("reroute_count", 0) + 1,
            "last_used": datetime.utcnow().isoformat(),
            "last_delay_mins": context.get("delay_mins", 0),
            "last_line": line_id,
            "avg_cost": round(
                (current.get("avg_cost", 0) *
                current.get("reroute_count", 0) +
                float(best_supplier["cost"])) /
                (current.get("reroute_count", 0) + 1), 2
            )
        })
        print(f"[{self.agent_name}] Belief updated: "
            f"{supplier_id} reroute count = "
            f"{current.get('reroute_count', 0) + 1}")


        # Notify inventory agent to execute reroute
        await self.publish("BUFFER_ALERT", {
            "line_id": line_id,
            "order_id": order_id,
            "severity": "CRITICAL",
            "action": "REROUTE_TO_BACKUP",
            "backup_supplier_id": best_supplier["supplier_id"],
            "backup_supplier_name": best_supplier["name"],
            "backup_reliability": best_supplier["reliability"],
            "delay_mins": delay_mins,
            "neural_confidence": prob_late,
            "part_type": part_type
        })

        return {
            "action": "REROUTE_TO_BACKUP",
            "order_id": order_id,
            "backup_supplier": best_supplier["supplier_id"]
        }