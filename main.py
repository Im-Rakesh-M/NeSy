import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import asyncio
from datetime import datetime
from core.neo4j_driver import Neo4jConnection
from agents.message_bus import MessageBus
from agents.logistics_agent import LogisticsAgent
from agents.maintenance_agent import MaintenanceAgent
from agents.inventory_agent import InventoryAgent
from agents.hr_agent import HRAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.sop_agent import SOPAgent
from agents.bom_agent import BOMAgent
from neural.delivery_forecaster import DeliveryForecaster
from neural.machine_forecaster import MachineForecaster
from agents.bom_agent import BOMAgent


class NESODigitalTwin:
    """
    NESO-DT Master Orchestrator.

    Responsibilities:
    1. Load pre-trained neural models
    2. Initialize message bus
    3. Instantiate all agents
    4. Run Digital Twin state sync loop
    5. Process events from Knowledge Graph
    6. Coordinate agent decisions via message bus

    Rationale for event-driven loop:
    A Digital Twin must continuously mirror the physical
    system state. The sync loop reads new events from
    Neo4j, publishes them to the message bus, and lets
    agents react autonomously — no central controller
    tells agents what to do.
    """

    def __init__(self):
        print("=" * 60)
        print("   NESO-DT DIGITAL TWIN INITIALIZING")
        print("=" * 60)

        # Initialize connection
        self.conn = Neo4jConnection.get_instance()

        # Load pre-trained models
        print("\n[NESO-DT] Loading neural models...")
        self.delivery_forecaster = DeliveryForecaster()
        self.delivery_forecaster.load()

        self.machine_forecaster = MachineForecaster()
        self.machine_forecaster.load()
        print("[NESO-DT] Neural models loaded.")

        # Initialize message bus
        self.bus = MessageBus()

        # Instantiate agents
        print("\n[NESO-DT] Initializing agents...")
        self.logistics = LogisticsAgent(self.bus)
        self.maintenance = MaintenanceAgent(self.bus)
        self.inventory = InventoryAgent(self.bus)
        self.hr = HRAgent(self.bus)
        self.regulatory = RegulatoryAgent(self.bus)
        self.sop = SOPAgent(self.bus)
        self.bom = BOMAgent(self.bus)
        print("[NESO-DT] All agents online.")
        print(f"[NESO-DT] Message bus topics: "
              f"{self.bus.get_topic_stats()}")

    async def process_delivery_event(self, order: dict):
        """
        Processes a single delivery order through
        the neural forecaster and publishes result
        to message bus for agent processing.
        """
        features = {
            "urgency_score": order.get("urgency_score", 0),
            "Order Item Quantity": order.get("quantity", 0),
            "Order Region": order.get("region", ""),
            "Shipping Mode": order.get("shipping_mode", ""),
            "Category Id": order.get("category_id", 0),
            "delay_days": order.get("delay_days", 0.0)
        }

        result = self.delivery_forecaster.predict(features)

        await self.bus.publish("DELIVERY_RISK", {
            "order_id": order.get("order_id"),
            "line_id": order.get("line_id"),
            "part_type": order.get("part_type"),
            "region": order.get("region"),
            "late_probability": result["late_probability"],
            "prediction_set": result["prediction_set"],
            "confidence": result["confidence"],
            "will_be_late": result["will_be_late"],
            "delay_mins": order.get("delay_mins", 0.0),
            "sender": "NESO_ORCHESTRATOR"
        })

    async def process_machine_event(self, machine: dict):
        """
        Processes a single machine state through
        the neural forecaster and publishes result
        to message bus for agent processing.
        """
        features = {
            "thermal_stress": machine.get("thermal_stress", 0.0),
            "mechanical_load": machine.get("mechanical_load", 0.0),
            "wear_criticality": machine.get("wear_criticality", 0.0),
            "failure_risk_score": machine.get("failure_risk_score", 0),
            "jit_supply_pressure": machine.get("jit_supply_pressure", 0.0),
            "product_type_encoded": machine.get("product_type_encoded", 0)
        }

        result = self.machine_forecaster.predict(features)

        await self.bus.publish("MACHINE_RISK", {
            "machine_id": machine.get("uid"),
            "line_id": machine.get("line_id", "UNKNOWN"),
            "failure_probability": result["failure_probability"],
            "prediction_set": result["prediction_set"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "will_fail": result["will_fail"],
            "wear_criticality": machine.get("wear_criticality", 0.0),
            "thermal_stress": machine.get("thermal_stress", 0.0),
            "operation_hours": machine.get("tool_wear", 0),
            "sender": "NESO_ORCHESTRATOR"
        })

    async def run_sync_loop(
        self,
        delivery_limit=20,
        machine_limit=20
    ):
        """
        Digital Twin state synchronisation loop.

        Fetches recent high-risk events from Neo4j
        and processes them through the agent network.

        Rationale for limit parameters:
        In a real deployment this runs continuously.
        For dissertation demo we process a controlled
        batch to show the full decision chain clearly.
        """
        print(f"\n[NESO-DT] Starting state sync loop...")
        print(f"  -> Processing {delivery_limit} delivery events")
        print(f"  -> Processing {machine_limit} machine events")
        print("-" * 60)

        # Fetch high-risk delivery orders from graph
        delivery_results = []
        for line in ['LINE_A', 'LINE_B', 'LINE_C', 'LINE_D']:
            with self.conn.session() as session:
                results = session.run("""
                    MATCH (d:DeliveryOrder)-[:DELIVERS_TO]->(l:ProductionLine {line_id: $line_id})
                    WHERE d.risk_level IN ['MODERATE_DELAY', 'CRITICAL_DELAY']
                    RETURN d.order_id AS order_id,
                        d.delay_days AS delay_days,
                        d.delay_mins AS delay_mins,
                        d.urgency_score AS urgency_score,
                        d.quantity AS quantity,
                        d.region AS region,
                        d.shipping_mode AS shipping_mode,
                        d.part_type AS part_type,
                        d.line_id AS line_id,
                        l.line_id AS confirmed_line
                    LIMIT $limit
                """, line_id=line, limit=delivery_limit // 4)
                delivery_results.extend([dict(r) for r in results])

        delivery_list = delivery_results
        print(f"[NESO-DT] Fetched {len(delivery_list)} "
            f"at-risk delivery events from graph")


        # Fetch high-risk machine states from graph
        machine_results = []
        for line in ['LINE_A', 'LINE_B', 'LINE_C', 'LINE_D']:
            with self.conn.session() as session:
                results = session.run("""
                    MATCH (m:ProductionHour)-[:OPERATES_ON]->(l:ProductionLine {line_id: $line_id})
                    WHERE m.compliance_status = 'VIOLATION_DETECTED'
                    RETURN m.uid AS uid,
                        m.thermal_stress AS thermal_stress,
                        m.mechanical_load AS mechanical_load,
                        m.wear_criticality AS wear_criticality,
                        m.failure_risk_score AS failure_risk_score,
                        m.jit_supply_pressure AS jit_supply_pressure,
                        m.product_type_encoded AS product_type_encoded,
                        m.tool_wear AS tool_wear,
                        l.line_id AS line_id
                    LIMIT $limit
                """, line_id=line, limit=machine_limit // 4)
                machine_results.extend([dict(r) for r in results])

        machine_list = machine_results
        print(f"[NESO-DT] Fetched {len(machine_list)} "
            f"at-risk machine states from graph")

        # Process delivery events
        print(f"\n[NESO-DT] Processing delivery events...")
        for i, order in enumerate(delivery_list):
            print(f"\n--- Delivery Event {i+1}/{len(delivery_list)} ---")
            await self.process_delivery_event(order)
            await asyncio.sleep(0.1)

        # Process machine events
        print(f"\n[NESO-DT] Processing machine events...")
        for i, machine in enumerate(machine_list):
            print(f"\n--- Machine Event {i+1}/{len(machine_list)} ---")
            await self.process_machine_event(machine)
            await asyncio.sleep(0.1)

        # Print final audit summary
        await self.print_audit_summary()

    async def print_audit_summary(self):
        """Queries Neo4j for audit trail summary."""
        print("\n" + "=" * 60)
        print("   NESO-DT AUDIT TRAIL SUMMARY")
        print("=" * 60)

        with self.conn.session() as session:
            results = session.run("""
                MATCH (a:AuditEntry)
                RETURN a.agent_name AS agent,
                       a.compliance_status AS status,
                       count(a) AS decisions
                ORDER BY decisions DESC
            """)
            records = list(results)

        if not records:
            print("No audit entries found.")
            return

        for r in records:
            print(f"  {r['agent']}: "
                  f"{r['decisions']} decisions | "
                  f"Status: {r['status']}")

        # Total cost of all mitigations
        with self.conn.session() as session:
            cost = session.run("""
                MATCH (a:AuditEntry)
                WHERE a.cost_of_action > 0
                RETURN sum(a.cost_of_action) AS total_cost,
                       count(a) AS paid_actions
            """).single()

        if cost:
            print(f"\n  Total mitigation cost : "
                  f"€{cost['total_cost']:,.2f}")
            print(f"  Paid actions          : "
                  f"{cost['paid_actions']}")
            
        # Print belief store summary
        with self.conn.session() as session:
            beliefs = session.run("""
                MATCH (b:AgentBelief)
                RETURN b.agent_name AS agent,
                    b.key AS key,
                    b.value AS value,
                    b.updated AS updated
                ORDER BY b.agent_name, b.key
            """)
            belief_records = list(beliefs)

        if belief_records:
            print(f"\n  Agent Belief Store ({len(belief_records)} beliefs):")
            print(f"  {'-'*40}")
            for b in belief_records:
                print(f"  [{b['agent']}] {b['key']}")
                try:
                    val = eval(b['value'])
                    for k, v in val.items():
                        print(f"    {k}: {v}")
                except Exception:
                    print(f"    {b['value']}")
                print()

 # BOM completion risk summary
        with self.conn.session() as session:
            bom_risks = session.run("""
                MATCH (v:Vehicle)-[r:AT_RISK]->(b:BOMItem)
                      -[:FULFILLED_BY]->(l:ProductionLine)
                RETURN v.model_id AS model,
                       v.daily_target AS daily_target,
                       b.part_type AS blocked_part,
                       l.line_id AS line_id,
                       r.severity AS severity,
                       r.wip_cost_eur AS wip_cost
                ORDER BY r.severity DESC
            """)
            bom_records = list(bom_risks)

        if bom_records:
            total_blocked = sum(r["daily_target"] for r in bom_records)
            total_wip = sum(r["wip_cost"] or 0 for r in bom_records)
            print(f"\n  BOM Completion Risk:")
            print(f"  {'-'*40}")
            for r in bom_records:
                print(f"  {r['model']} | {r['blocked_part']} "
                      f"| {r['severity']} | "
                      f"WIP: €{r['wip_cost']:,}")
            print(f"\n  Total vehicles at risk : {total_blocked}")
            print(f"  Total WIP cost         : €{total_wip:,}")
        else:
            print(f"\n  BOM Completion Risk: No vehicles at risk")

async def main():
    twin = NESODigitalTwin()
    await twin.run_sync_loop(
        delivery_limit=20,
        machine_limit=20
    )


if __name__ == "__main__":
    asyncio.run(main())