import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from core.neo4j_driver import Neo4jConnection

np.random.seed(42)
random.seed(42)

# ================================================================
# DOMAIN 1 — PRODUCTION LINES
# Rationale: A typical automotive OEM plant has 4-8 assembly lines
# each dedicated to specific vehicle subsystems. We model 4 lines
# covering the major JIT-sensitive subsystems.
# ================================================================

PRODUCTION_LINES = [
    {
        "line_id": "LINE_A",
        "name": "Engine Assembly Line",
        "part_type": "ENGINE_BLOCK",
        "buffer_units": 45,
        "buffer_capacity": 100,
        "safety_stock_units": 20,
        "cycle_time_mins": 12,
        "shift_requirement": 4,
        "certified_role": "ENGINE_TECHNICIAN"
    },
    {
        "line_id": "LINE_B",
        "name": "Transmission Assembly Line",
        "part_type": "TRANSMISSION",
        "buffer_units": 30,
        "buffer_capacity": 80,
        "safety_stock_units": 15,
        "cycle_time_mins": 18,
        "shift_requirement": 3,
        "certified_role": "TRANSMISSION_TECHNICIAN"
    },
    {
        "line_id": "LINE_C",
        "name": "Body Panel Line",
        "part_type": "DOOR_PANEL",
        "buffer_units": 120,
        "buffer_capacity": 200,
        "safety_stock_units": 40,
        "cycle_time_mins": 6,
        "shift_requirement": 5,
        "certified_role": "BODY_TECHNICIAN"
    },
    {
        "line_id": "LINE_D",
        "name": "Chassis Frame Line",
        "part_type": "CHASSIS_FRAME",
        "buffer_units": 25,
        "buffer_capacity": 60,
        "safety_stock_units": 10,
        "cycle_time_mins": 24,
        "shift_requirement": 3,
        "certified_role": "CHASSIS_TECHNICIAN"
    }
]

# ================================================================
# DOMAIN 2 — SUPPLIERS
# Rationale: Each part type has one primary and two backup suppliers.
# Backup suppliers have higher cost but guaranteed availability.
# Reliability scores derived from DataCo regional delivery performance.
# ================================================================

SUPPLIERS = [
    # ENGINE_BLOCK suppliers
    {
        "supplier_id": "SUP_001",
        "name": "Magna Powertrain GmbH",
        "part_type": "ENGINE_BLOCK",
        "supplier_tier": "PRIMARY",
        "region": "Western Europe",
        "reliability_score": 0.94,
        "lead_time_days": 2,
        "cost_per_unit": 8500,
        "min_order_qty": 10
    },
    {
        "supplier_id": "SUP_002",
        "name": "BorgWarner Engine Systems",
        "part_type": "ENGINE_BLOCK",
        "supplier_tier": "BACKUP_1",
        "region": "East of USA",
        "reliability_score": 0.88,
        "lead_time_days": 3,
        "cost_per_unit": 9200,
        "min_order_qty": 5
    },
    {
        "supplier_id": "SUP_003",
        "name": "Aisin Engine Components",
        "part_type": "ENGINE_BLOCK",
        "supplier_tier": "BACKUP_2",
        "region": "Eastern Asia",
        "reliability_score": 0.85,
        "lead_time_days": 5,
        "cost_per_unit": 7800,
        "min_order_qty": 20
    },
    # TRANSMISSION suppliers
    {
        "supplier_id": "SUP_004",
        "name": "ZF Friedrichshafen AG",
        "part_type": "TRANSMISSION",
        "supplier_tier": "PRIMARY",
        "region": "Western Europe",
        "reliability_score": 0.96,
        "lead_time_days": 2,
        "cost_per_unit": 6200,
        "min_order_qty": 8
    },
    {
        "supplier_id": "SUP_005",
        "name": "Allison Transmission",
        "part_type": "TRANSMISSION",
        "supplier_tier": "BACKUP_1",
        "region": "West of USA",
        "reliability_score": 0.91,
        "lead_time_days": 4,
        "cost_per_unit": 6800,
        "min_order_qty": 5
    },
    {
        "supplier_id": "SUP_006",
        "name": "Jatco Transmission Systems",
        "part_type": "TRANSMISSION",
        "supplier_tier": "BACKUP_2",
        "region": "Eastern Asia",
        "reliability_score": 0.87,
        "lead_time_days": 6,
        "cost_per_unit": 5900,
        "min_order_qty": 15
    },
    # DOOR_PANEL suppliers
    {
        "supplier_id": "SUP_007",
        "name": "Gestamp Automocion",
        "part_type": "DOOR_PANEL",
        "supplier_tier": "PRIMARY",
        "region": "Southern Europe",
        "reliability_score": 0.92,
        "lead_time_days": 1,
        "cost_per_unit": 1200,
        "min_order_qty": 50
    },
    {
        "supplier_id": "SUP_008",
        "name": "Magna Exteriors",
        "part_type": "DOOR_PANEL",
        "supplier_tier": "BACKUP_1",
        "region": "Western Europe",
        "reliability_score": 0.95,
        "lead_time_days": 2,
        "cost_per_unit": 1350,
        "min_order_qty": 30
    },
    {
        "supplier_id": "SUP_009",
        "name": "Gestamp India",
        "part_type": "DOOR_PANEL",
        "supplier_tier": "BACKUP_2",
        "region": "South Asia",
        "reliability_score": 0.83,
        "lead_time_days": 3,
        "cost_per_unit": 980,
        "min_order_qty": 100
    },
    # CHASSIS_FRAME suppliers
    {
        "supplier_id": "SUP_010",
        "name": "Thyssenkrupp Chassis",
        "part_type": "CHASSIS_FRAME",
        "supplier_tier": "PRIMARY",
        "region": "Western Europe",
        "reliability_score": 0.93,
        "lead_time_days": 3,
        "cost_per_unit": 4500,
        "min_order_qty": 5
    },
    {
        "supplier_id": "SUP_011",
        "name": "Martinrea International",
        "part_type": "CHASSIS_FRAME",
        "supplier_tier": "BACKUP_1",
        "region": "US Center",
        "reliability_score": 0.89,
        "lead_time_days": 4,
        "cost_per_unit": 4900,
        "min_order_qty": 5
    },
    {
        "supplier_id": "SUP_012",
        "name": "Tata AutoComp Chassis",
        "part_type": "CHASSIS_FRAME",
        "supplier_tier": "BACKUP_2",
        "region": "South Asia",
        "reliability_score": 0.86,
        "lead_time_days": 5,
        "cost_per_unit": 4100,
        "min_order_qty": 10
    }
]

# ================================================================
# DOMAIN 3 — SHIFT WORKERS
# Rationale: Three 8-hour shifts covering 24hr operations.
# Each worker is certified for specific line roles.
# Availability is probabilistic — 85% baseline attendance rate
# reflecting real automotive plant absenteeism data.
# ================================================================

ROLES = [
    "ENGINE_TECHNICIAN",
    "TRANSMISSION_TECHNICIAN",
    "BODY_TECHNICIAN",
    "CHASSIS_TECHNICIAN",
    "QUALITY_INSPECTOR",
    "LOGISTICS_COORDINATOR"
]

SHIFTS = [
    {"shift_id": "SHIFT_1", "start": "06:00", "end": "14:00"},
    {"shift_id": "SHIFT_2", "start": "14:00", "end": "22:00"},
    {"shift_id": "SHIFT_3", "start": "22:00", "end": "06:00"}
]

def generate_shift_workers(n=60):
    """
    Generates 60 workers across 3 shifts and 4 lines.
    20 workers per shift — realistic for a 4-line OEM plant.
    Each worker has one primary certification and one secondary.
    Availability follows a Bernoulli distribution with p=0.85.
    """
    workers = []
    worker_id = 1

    for shift in SHIFTS:
        for i in range(20):
            primary_role = random.choice(ROLES)
            secondary_role = random.choice(
                [r for r in ROLES if r != primary_role]
            )
            workers.append({
                "worker_id": f"W{worker_id:03d}",
                "name": f"Worker_{worker_id:03d}",
                "shift_id": shift["shift_id"],
                "shift_start": shift["start"],
                "shift_end": shift["end"],
                "primary_role": primary_role,
                "secondary_role": secondary_role,
                "experience_years": round(random.uniform(1, 20), 1),
                "availability": int(np.random.binomial(1, 0.85)),
                "overtime_eligible": random.choice([True, False]),
                "last_training_date": (
                    datetime.now() - timedelta(days=random.randint(1, 365))
                ).strftime("%Y-%m-%d")
            })
            worker_id += 1

    return pd.DataFrame(workers)

# ================================================================
# DOMAIN 4 — REGULATORY RULES
# Rationale: Automotive OEM plants operate under multiple
# overlapping regulatory frameworks. We model the three most
# relevant to JIT operations in India and EU markets.
# ================================================================

REGULATORY_RULES = [
    {
        "rule_id": "REG_001",
        "name": "EU Machinery Directive 2006/42/EC",
        "description": "Maximum continuous machine operation without maintenance inspection",
        "jurisdiction": "European Union",
        "max_operation_hours": 90,
        "mandatory_downtime_mins": 30,
        "penalty_eur": 50000,
        "applies_to": "MachineUnit",
        "hard_constraint": True
    },
    {
        "rule_id": "REG_002",
        "name": "India Factories Act 1948 — Section 51",
        "description": "Maximum consecutive working hours for shift workers without break",
        "jurisdiction": "India",
        "max_consecutive_hours": 9,
        "mandatory_break_mins": 30,
        "penalty_eur": 5000,
        "applies_to": "ShiftWorker",
        "hard_constraint": True
    },
    {
        "rule_id": "REG_003",
        "name": "ISO 45001:2018 Occupational Health",
        "description": "Minimum certified operators required per active production line",
        "jurisdiction": "Global",
        "min_certified_operators": 2,
        "penalty_eur": 25000,
        "applies_to": "ProductionLine",
        "hard_constraint": True
    },
    {
        "rule_id": "REG_004",
        "name": "EU REACH Regulation EC 1907/2006",
        "description": "Hazardous material handling — certified operator mandatory",
        "jurisdiction": "European Union",
        "requires_certification": "HAZMAT_CERTIFIED",
        "penalty_eur": 75000,
        "applies_to": "ShiftWorker",
        "hard_constraint": True
    },
    {
        "rule_id": "REG_005",
        "name": "ISO 9001:2015 Quality Management",
        "description": "Quality inspection mandatory before parts enter production line",
        "jurisdiction": "Global",
        "requires_role": "QUALITY_INSPECTOR",
        "penalty_eur": 15000,
        "applies_to": "DeliveryOrder",
        "hard_constraint": False
    }
]

# ================================================================
# NEO4J SEEDING FUNCTIONS
# ================================================================

def seed_production_lines(conn):
    print("[SYNTHETIC] Seeding production lines...")
    with conn.session() as session:
        session.run("MATCH (n:ProductionLine) DETACH DELETE n")
    
    for line in PRODUCTION_LINES:
        with conn.session() as session:
            session.run("""
                CREATE (:ProductionLine {
                    line_id: $line_id,
                    name: $name,
                    part_type: $part_type,
                    buffer_units: $buffer_units,
                    buffer_capacity: $buffer_capacity,
                    safety_stock_units: $safety_stock_units,
                    buffer_pct: $buffer_pct,
                    cycle_time_mins: $cycle_time_mins,
                    shift_requirement: $shift_requirement,
                    certified_role: $certified_role,
                    status: 'OPERATIONAL'
                })
            """, **line,
            buffer_pct=round(
                line['buffer_units'] / line['buffer_capacity'] * 100, 1
            ))
    print(f"  -> {len(PRODUCTION_LINES)} production lines seeded.")

def seed_suppliers(conn):
    print("[SYNTHETIC] Seeding suppliers...")
    with conn.session() as session:
        session.run("MATCH (n:Supplier) DETACH DELETE n")
    
    for sup in SUPPLIERS:
        with conn.session() as session:
            session.run("""
                CREATE (:Supplier {
                    supplier_id: $supplier_id,
                    name: $name,
                    part_type: $part_type,
                    supplier_tier: $supplier_tier,
                    region: $region,
                    reliability_score: $reliability_score,
                    lead_time_days: $lead_time_days,
                    cost_per_unit: $cost_per_unit,
                    min_order_qty: $min_order_qty,
                    status: 'ACTIVE'
                })
            """, **sup)
    print(f"  -> {len(SUPPLIERS)} suppliers seeded.")

def seed_shift_workers(conn, df_workers):
    print("[SYNTHETIC] Seeding shift workers...")
    with conn.session() as session:
        session.run("MATCH (n:ShiftWorker) DETACH DELETE n")

    records = df_workers.to_dict('records')
    with conn.session() as session:
        session.run("""
            UNWIND $batch AS row
            CREATE (:ShiftWorker {
                worker_id: row.worker_id,
                name: row.name,
                shift_id: row.shift_id,
                shift_start: row.shift_start,
                shift_end: row.shift_end,
                primary_role: row.primary_role,
                secondary_role: row.secondary_role,
                experience_years: row.experience_years,
                availability: row.availability,
                overtime_eligible: row.overtime_eligible,
                last_training_date: row.last_training_date
            })
        """, batch=records)
    print(f"  -> {len(records)} shift workers seeded.")

def seed_regulatory_rules(conn):
    print("[SYNTHETIC] Seeding regulatory rules...")
    with conn.session() as session:
        session.run("MATCH (n:RegulatoryRule) DETACH DELETE n")

    for rule in REGULATORY_RULES:
        with conn.session() as session:
            session.run("""
                CREATE (:RegulatoryRule {
                    rule_id: $rule_id,
                    name: $name,
                    description: $description,
                    jurisdiction: $jurisdiction,
                    penalty_eur: $penalty_eur,
                    applies_to: $applies_to,
                    hard_constraint: $hard_constraint
                })
            """, **{k: v for k, v in rule.items()
                   if k not in ['max_operation_hours',
                                'mandatory_downtime_mins',
                                'max_consecutive_hours',
                                'mandatory_break_mins',
                                'min_certified_operators',
                                'requires_certification',
                                'requires_role']})
    print(f"  -> {len(REGULATORY_RULES)} regulatory rules seeded.")

def seed_relationships(conn):
    """
    Creates all cross-domain relationships in the graph.
    Rationale: Relationships are the core value of a KG —
    they enable graph traversal queries that no relational
    database can match for speed and expressiveness.
    """
    print("[SYNTHETIC] Creating cross-domain relationships...")

    with conn.session() as session:
        # Supplier → ProductionLine (via part type)
        session.run("""
            MATCH (s:Supplier), (l:ProductionLine)
            WHERE s.part_type = l.part_type
            AND s.supplier_tier = 'PRIMARY'
            CREATE (s)-[:PRIMARY_SUPPLIER_FOR]->(l)
        """)
        session.run("""
            MATCH (s:Supplier), (l:ProductionLine)
            WHERE s.part_type = l.part_type
            AND s.supplier_tier IN ['BACKUP_1', 'BACKUP_2']
            CREATE (s)-[:BACKUP_SUPPLIER_FOR {
                tier: s.supplier_tier
            }]->(l)
        """)
        print("  -> Supplier → ProductionLine relationships created.")

        # ShiftWorker → ProductionLine (via certified role)
        session.run("""
            MATCH (w:ShiftWorker), (l:ProductionLine)
            WHERE w.primary_role = l.certified_role
            CREATE (w)-[:CERTIFIED_FOR]->(l)
        """)
        print("  -> ShiftWorker → ProductionLine relationships created.")

        # ProductionLine → SOPRule
        session.run("""
            MATCH (l:ProductionLine), (s:SOPRule)
            CREATE (l)-[:GOVERNED_BY]->(s)
        """)
        print("  -> ProductionLine → SOPRule relationships created.")

        # ProductionLine → RegulatoryRule
        session.run("""
            MATCH (l:ProductionLine), (r:RegulatoryRule)
            WHERE r.applies_to IN ['ProductionLine', 'MachineUnit',
                                   'DeliveryOrder']
            CREATE (l)-[:MUST_COMPLY_WITH]->(r)
        """)
        print("  -> ProductionLine → RegulatoryRule relationships created.")

        # MachineUnit → ProductionLine (assign machines to lines)
        session.run("""
            MATCH (m:ProductionHour), (l:ProductionLine)
            WHERE m.uid % 4 = 
                CASE l.line_id
                    WHEN 'LINE_A' THEN 0
                    WHEN 'LINE_B' THEN 1
                    WHEN 'LINE_C' THEN 2
                    WHEN 'LINE_D' THEN 3
                END
            CREATE (m)-[:OPERATES_ON]->(l)
        """)
        print("  -> MachineUnit → ProductionLine relationships created.")

def run_generator():
    print("=" * 60)
    print("   NESO-DT SYNTHETIC DATA GENERATOR")
    print("=" * 60)

    conn = Neo4jConnection.get_instance()

    # Generate and seed all domains
    seed_production_lines(conn)
    seed_suppliers(conn)

    df_workers = generate_shift_workers(n=60)
    df_workers.to_csv("data/synthetic_workers.csv", index=False)
    seed_shift_workers(conn, df_workers)

    seed_regulatory_rules(conn)
    seed_relationships(conn)

    # Verify full graph state
    with conn.session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        print("\n[VERIFY] Complete Knowledge Graph Node Count:")
        print("-" * 40)
        for r in result:
            print(f"  {r['label']}: {r['count']} nodes")

    with conn.session() as session:
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS relationship, count(r) AS count
            ORDER BY count DESC
        """)
        print("\n[VERIFY] Relationship Summary:")
        print("-" * 40)
        for r in result:
            print(f"  {r['relationship']}: {r['count']}")

    print("\n[SUCCESS] Digital Twin knowledge graph fully populated.")
    conn.close()

if __name__ == "__main__":
    run_generator()