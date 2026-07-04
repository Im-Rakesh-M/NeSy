import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from agents.base_agent import BaseAgent


class BOMAgent(BaseAgent):
    """
    Monitors vehicle completion risk using Bill of Materials.

    For each vehicle on the build schedule, checks whether
    all required BOM items can be supplied given:
    - Current delivery delay risk per production line
    - Current machine health per production line
    - Current buffer levels per production line

    When a BOM item is at risk, creates AT_RISK relationship
    in the Knowledge Graph and calculates:
    - How many vehicles are blocked
    - WIP cost of completed sub-assemblies waiting
    - Recommended build resequencing

    Subscribes to: CASCADE_ALERT, BUFFER_ALERT, MACHINE_RISK
    Publishes to: CASCADE_ALERT (when vehicles blocked)
    """

    def __init__(self, message_bus):
        super().__init__("BOMAgent", message_bus)
        # Subscribe only to CASCADE_ALERT — this fires when
        # severity is confirmed critical, not on every event.
        # Avoids BOMAgent firing multiple times per machine event.
        self.bus.subscribe("CASCADE_ALERT", self.perceive)

    async def perceive(self, payload: Dict[str, Any]):
        """Receives supply chain event and assesses BOM impact."""
        line_id = payload.get("line_id")
        if not line_id:
            return
        await self.decide({
                    "line_id": line_id,
                    "severity": payload.get("severity", "LOW"),
                    "topic": payload.get("_topic", "UNKNOWN"),
                    "neural_confidence": payload.get(
                        "neural_confidence",
                        payload.get("failure_probability", 0.0)
                    ),
                    "upstream_neural_confidence": payload.get(
                        "neural_confidence",
                        payload.get("failure_probability", 0.0)
                    )
                })

    async def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assesses vehicle completion risk when a line is disrupted.

        Logic:
        1. Find which BOM items are sourced from disrupted line
        2. Find which vehicles require those BOM items
        3. Calculate how many vehicles are blocked
        4. Calculate WIP cost (completed parts waiting)
        5. Create AT_RISK relationships in graph
        6. Suggest resequencing if possible
        """
        line_id = context["line_id"]
        severity = context["severity"]
        confidence = context["neural_confidence"]

        print(f"[{self.agent_name}] Assessing BOM impact for "
              f"{line_id} | Severity={severity}")

        # Find BOM items sourced from this line
        bom_items = self.query_graph("""
            MATCH (b:BOMItem)-[:FULFILLED_BY]->(l:ProductionLine {line_id: $line_id})
            RETURN b.part_type AS part_type,
                   b.daily_requirement AS daily_req,
                   b.criticality AS criticality,
                   b.unit_cost_eur AS unit_cost
        """, {"line_id": line_id})

        if not bom_items:
            print(f"[{self.agent_name}] No BOM items linked to {line_id}")
            return {"action": "NO_BOM_IMPACT", "line_id": line_id}

        total_blocked = 0
        total_wip_cost = 0
        affected_models = []

        for item in bom_items:
            part_type = item["part_type"]
            criticality = item["criticality"]
            unit_cost = item["unit_cost"] or 0

            # Find vehicles that require this part
            vehicles = self.query_graph("""
                MATCH (v:Vehicle)-[r:REQUIRES]->(b:BOMItem {part_type: $part_type})
                RETURN v.model_id AS model_id,
                       v.name AS name,
                       v.daily_target AS daily_target,
                       r.quantity_per_vehicle AS qty_per_vehicle,
                       r.total_daily_qty AS total_daily_qty
            """, {"part_type": part_type})

            for vehicle in vehicles:
                model_id = vehicle["model_id"]
                daily_target = vehicle["daily_target"]
                total_daily = vehicle["total_daily_qty"]

                # Calculate WIP cost — other completed parts waiting
                wip_cost = self._calculate_wip_cost(
                    model_id, part_type, daily_target
                )
                total_blocked += daily_target
                total_wip_cost += wip_cost

                if model_id not in affected_models:
                    affected_models.append(model_id)

                print(f"  → {model_id}: {daily_target} vehicles blocked | "
                      f"WIP cost: €{wip_cost:,}")

                # Create AT_RISK relationship in graph
                self._create_at_risk_relationship(
                    model_id, part_type, severity, wip_cost, confidence
                )

        # Log audit
        self.log_audit(
            event_type="BOM_RISK_ASSESSMENT",
            decision=f"VEHICLES_AT_RISK_{line_id}",
            confidence=1.0,  # BOMAgent uses deterministic graph logic
            rule_fired="BOM_COMPLETION_CHECK",
            status="VIOLATION_DETECTED" if total_blocked > 0
                   else "COMPLIANT",
            cost=float(total_wip_cost),
            alternative="Resequence build order"
        )

        if total_blocked > 0:
            print(f"\n[{self.agent_name}] 🚨 BOM IMPACT SUMMARY:")
            print(f"  Line affected    : {line_id}")
            print(f"  Vehicles blocked : {total_blocked}")
            print(f"  Models affected  : {affected_models}")
            print(f"  Total WIP cost   : €{total_wip_cost:,}")

            # Check if resequencing is possible
            resequence = self._check_resequencing(
                affected_models, line_id
            )

            if resequence["possible"]:
                print(f"  Resequencing     : POSSIBLE → "
                      f"build {resequence['safe_models']} first")
            else:
                print(f"  Resequencing     : NOT POSSIBLE — "
                      f"all models require {line_id} parts")

                # Escalate if critical
                if severity == "CRITICAL":
                    await self.publish("CASCADE_ALERT", {
                        "line_id": line_id,
                        "severity": "CRITICAL",
                        "reason": "BOM_COMPLETION_RISK",
                        "vehicles_blocked": total_blocked,
                        "models_affected": str(affected_models),
                        "wip_cost_eur": total_wip_cost,
                        "action_required": "HALT_BUILD_SCHEDULE"
                    })

        return {
            "action": "BOM_ASSESSED",
            "line_id": line_id,
            "vehicles_blocked": total_blocked,
            "wip_cost": total_wip_cost,
            "affected_models": affected_models
        }

    def _calculate_wip_cost(
        self, model_id: str, missing_part: str, daily_target: int
    ) -> float:
        """
        Calculates WIP cost — value of completed sub-assemblies
        sitting idle waiting for the missing part.

        If TRANSMISSION is delayed, ENGINE_BLOCK, DOOR_PANEL,
        and CHASSIS_FRAME are already assembled and waiting.
        That idle inventory has a financial cost.
        """
        # Get all OTHER parts for this vehicle (already assembled)
        other_parts = self.query_graph("""
            MATCH (v:Vehicle {model_id: $model_id})
                  -[r:REQUIRES]->(b:BOMItem)
            WHERE b.part_type <> $missing_part
            RETURN b.part_type AS part_type,
                   b.unit_cost_eur AS unit_cost,
                   r.quantity_per_vehicle AS qty
        """, {"model_id": model_id, "missing_part": missing_part})

        wip_per_vehicle = sum(
            (p["unit_cost"] or 0) * (p["qty"] or 1)
            for p in other_parts
        )
        return wip_per_vehicle * daily_target

    def _create_at_risk_relationship(
        self, model_id: str, part_type: str,
        severity: str, wip_cost: float, confidence: float
    ):
        """Creates AT_RISK relationship in Knowledge Graph."""
        try:
            with self.conn.session() as session:
                session.run("""
                    MATCH (v:Vehicle {model_id: $model_id})
                    MATCH (b:BOMItem {part_type: $part_type})
                    MERGE (v)-[r:AT_RISK]->(b)
                    SET r.severity = $severity,
                        r.wip_cost_eur = $wip_cost,
                        r.neural_confidence = $confidence,
                        r.detected_at = datetime()
                """,
                model_id=model_id,
                part_type=part_type,
                severity=severity,
                wip_cost=wip_cost,
                confidence=confidence
                )
        except Exception as e:
            print(f"[{self.agent_name}] AT_RISK write failed: {e}")

    def _check_resequencing(
        self, affected_models: list, blocked_line: str
    ) -> Dict:
        """
        Checks if build schedule can be resequenced to
        build vehicles that don't need the blocked line's parts.

        In practice all 3 models need all 4 lines — but this
        method demonstrates the resequencing logic for the demo.
        """
        # Find models that DON'T need the blocked line
        safe_models = self.query_graph("""
            MATCH (v:Vehicle)
            WHERE NOT EXISTS {
                MATCH (v)-[:REQUIRES]->(b:BOMItem)
                      -[:FULFILLED_BY]->(l:ProductionLine {line_id: $line_id})
            }
            RETURN v.model_id AS model_id, v.daily_target AS target
        """, {"blocked_line": blocked_line})

        return {
            "possible": len(safe_models) > 0,
            "safe_models": [m["model_id"] for m in safe_models]
        }

    async def get_completion_risk_summary(self) -> Dict:
        """
        Returns current vehicle completion risk summary.
        Called by EQA engine for 'show vehicle risk' queries.
        """
        results = self.query_graph("""
            MATCH (v:Vehicle)-[r:AT_RISK]->(b:BOMItem)
                  -[:FULFILLED_BY]->(l:ProductionLine)
            RETURN v.model_id AS model,
                   v.daily_target AS daily_target,
                   b.part_type AS blocked_part,
                   l.line_id AS line_id,
                   r.severity AS severity,
                   r.wip_cost_eur AS wip_cost,
                   r.detected_at AS detected_at
            ORDER BY r.severity DESC, v.model_id
        """, {})

        return {
            "at_risk_vehicles": results,
            "total_blocked": sum(
                r["daily_target"] for r in results
            ),
            "total_wip_cost": sum(
                r["wip_cost"] or 0 for r in results
            )
        }