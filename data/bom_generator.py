import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from core.neo4j_driver import Neo4jConnection


# ── BOM Definition ─────────────────────────────────────────────────
# Top-level planning BOM for 3 vehicle models.
# Each entry: (part_type, quantity_per_vehicle, line_id, criticality)
# criticality: CRITICAL = line stops without it, HIGH = major impact

VEHICLE_BOMS = {
    "SEDAN_STD": {
        "name": "Standard Sedan",
        "daily_target": 45,
        "takt_time_mins": 12,
        "bom": [
            ("ENGINE_BLOCK",   1, "LINE_A", "CRITICAL"),
            ("TRANSMISSION",   1, "LINE_B", "CRITICAL"),
            ("DOOR_PANEL",     4, "LINE_C", "HIGH"),
            ("CHASSIS_FRAME",  1, "LINE_D", "CRITICAL"),
        ]
    },
    "SEDAN_SPT": {
        "name": "Sport Sedan",
        "daily_target": 20,
        "takt_time_mins": 15,
        "bom": [
            ("ENGINE_BLOCK",   1, "LINE_A", "CRITICAL"),
            ("TRANSMISSION",   1, "LINE_B", "CRITICAL"),
            ("DOOR_PANEL",     4, "LINE_C", "HIGH"),
            ("CHASSIS_FRAME",  1, "LINE_D", "CRITICAL"),
        ]
    },
    "SUV_STD": {
        "name": "Standard SUV",
        "daily_target": 15,
        "takt_time_mins": 18,
        "bom": [
            ("ENGINE_BLOCK",   1, "LINE_A", "CRITICAL"),
            ("TRANSMISSION",   1, "LINE_B", "CRITICAL"),
            ("DOOR_PANEL",     4, "LINE_C", "HIGH"),
            ("CHASSIS_FRAME",  1, "LINE_D", "CRITICAL"),
        ]
    }
}

# Daily part requirements derived from BOM x daily targets
# ENGINE_BLOCK : 45+20+15 = 80
# TRANSMISSION : 45+20+15 = 80
# DOOR_PANEL   : (45+20+15) x 4 = 320

def seed_bom():
    conn = Neo4jConnection.get_instance()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    build_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

    print("[BOM] Clearing existing BOM nodes...")
    with conn.session() as session:
        session.run("MATCH (n:Vehicle) DETACH DELETE n")
        session.run("MATCH (n:BOMItem) DETACH DELETE n")
        session.run("MATCH (n:BuildSchedule) DETACH DELETE n")
    print("[BOM] Cleared.")

    # Step 1: BuildSchedule
    with conn.session() as session:
        session.run("""
            CREATE (:BuildSchedule {
                schedule_id   : 'BS_001',
                build_date    : $build_date,
                created_date  : $today,
                total_vehicles: 80,
                sedan_std     : 45,
                sedan_spt     : 20,
                suv_std       : 15,
                status        : 'ACTIVE'
            })
        """, build_date=build_date, today=today)
    print("[BOM] BuildSchedule BS_001 created.")

    # Step 2: Calculate daily requirements
    part_requirements = {}
    for model_id, model in VEHICLE_BOMS.items():
        for part_type, qty, line_id, criticality in model["bom"]:
            if part_type not in part_requirements:
                part_requirements[part_type] = {
                    "line_id": line_id,
                    "criticality": criticality,
                    "total_daily_qty": 0
                }
            part_requirements[part_type]["total_daily_qty"] += (
                qty * model["daily_target"]
            )

    # Step 3: BOMItem nodes
    unit_costs = {
        "ENGINE_BLOCK":   8500,
        "TRANSMISSION":   6200,
        "DOOR_PANEL":     1200,
        "CHASSIS_FRAME":  4500
    }
    for part_type, details in part_requirements.items():
        with conn.session() as session:
            session.run("""
                CREATE (:BOMItem {
                    part_type        : $part_type,
                    part_number      : $part_number,
                    description      : $description,
                    daily_requirement: $daily_req,
                    line_id          : $line_id,
                    criticality      : $criticality,
                    unit_cost_eur    : $unit_cost
                })
            """,
            part_type=part_type,
            part_number=f"PN-{part_type[:3]}-001",
            description=part_type.replace("_", " ").title(),
            daily_req=details["total_daily_qty"],
            line_id=details["line_id"],
            criticality=details["criticality"],
            unit_cost=unit_costs.get(part_type, 1000)
            )
    print(f"[BOM] {len(part_requirements)} BOMItem nodes created.")

    # Step 4: Vehicle nodes
    for model_id, model in VEHICLE_BOMS.items():
        with conn.session() as session:
            session.run("""
                CREATE (:Vehicle {
                    model_id       : $model_id,
                    name           : $name,
                    daily_target   : $daily_target,
                    takt_time_mins : $takt_time_mins,
                    build_date     : $build_date,
                    completion_risk: 'LOW',
                    blocked_count  : 0
                })
            """,
            model_id=model_id,
            name=model["name"],
            daily_target=model["daily_target"],
            takt_time_mins=model["takt_time_mins"],
            build_date=build_date
            )
    print(f"[BOM] {len(VEHICLE_BOMS)} Vehicle nodes created.")

    # Step 5: REQUIRES relationships
    for model_id, model in VEHICLE_BOMS.items():
        for part_type, qty, line_id, criticality in model["bom"]:
            with conn.session() as session:
                session.run("""
                    MATCH (v:Vehicle {model_id: $model_id})
                    MATCH (b:BOMItem {part_type: $part_type})
                    CREATE (v)-[:REQUIRES {
                        quantity_per_vehicle: $qty,
                        total_daily_qty     : $total,
                        criticality         : $criticality
                    }]->(b)
                """,
                model_id=model_id,
                part_type=part_type,
                qty=qty,
                total=qty * model["daily_target"],
                criticality=criticality
                )
    print("[BOM] REQUIRES relationships created.")

    # Step 6: FULFILLED_BY relationships
    for part_type, details in part_requirements.items():
        with conn.session() as session:
            session.run("""
                MATCH (b:BOMItem {part_type: $part_type})
                MATCH (l:ProductionLine {line_id: $line_id})
                CREATE (b)-[:FULFILLED_BY]->(l)
            """,
            part_type=part_type,
            line_id=details["line_id"]
            )
    print("[BOM] FULFILLED_BY relationships created.")

    # Step 7: SCHEDULED_FOR relationships
    with conn.session() as session:
        session.run("""
            MATCH (v:Vehicle)
            MATCH (s:BuildSchedule {schedule_id: 'BS_001'})
            CREATE (v)-[:SCHEDULED_FOR]->(s)
        """)
    print("[BOM] SCHEDULED_FOR relationships created.")

    # Step 8: Link BOMItem to Suppliers
    with conn.session() as session:
        session.run("""
            MATCH (b:BOMItem)
            MATCH (s:Supplier)-[:PRIMARY_SUPPLIER_FOR]->(l:ProductionLine)
            WHERE l.line_id = b.line_id
            CREATE (b)-[:SOURCED_FROM {tier: 'PRIMARY'}]->(s)
        """)
        session.run("""
            MATCH (b:BOMItem)
            MATCH (s:Supplier)-[:BACKUP_SUPPLIER_FOR]->(l:ProductionLine)
            WHERE l.line_id = b.line_id
            CREATE (b)-[:SOURCED_FROM {tier: s.supplier_tier}]->(s)
        """)
    print("[BOM] SOURCED_FROM supplier relationships created.")

    # Verify
    with conn.session() as session:
        result = session.run("""
            MATCH (v:Vehicle)-[r:REQUIRES]->(b:BOMItem)
                  -[:FULFILLED_BY]->(l:ProductionLine)
            RETURN v.model_id AS model,
                   v.daily_target AS daily_target,
                   b.part_type AS part,
                   r.quantity_per_vehicle AS qty_per_vehicle,
                   r.total_daily_qty AS total_daily,
                   l.line_id AS line
            ORDER BY v.model_id, b.part_type
        """)
        records = list(result)
        print(f"\n[BOM] Verification — {len(records)} BOM lines:")
        print("-" * 65)
        for r in records:
            print(f"  {r['model']:<12} {r['part']:<16} "
                  f"qty/vehicle={r['qty_per_vehicle']} "
                  f"daily={r['total_daily']} "
                  f"line={r['line']}")

    print(f"\n[BOM] Knowledge Graph BOM integration complete.")
    conn.close()


if __name__ == "__main__":
    seed_bom()