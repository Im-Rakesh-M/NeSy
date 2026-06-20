from core.neo4j_driver import Neo4jConnection
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLEAR_QUERY = """
MATCH (n) WHERE n:SOPRule OR n:Mitigation OR n:FailureMode
DETACH DELETE n
"""

CREATE_QUERY = """
// ── Failure Mode Nodes ──────────────────────────────────────
CREATE (twf:FailureMode {
    code: 'TWF',
    name: 'Tool Wear Failure',
    description: 'Progressive tool degradation beyond safe threshold',
    automotive_analogue: 'Brake pad and drivetrain wear'
})
CREATE (hdf:FailureMode {
    code: 'HDF',
    name: 'Heat Dissipation Failure',
    description: 'Thermal management system breakdown',
    automotive_analogue: 'Coolant and thermal management failure'
})
CREATE (pwf:FailureMode {
    code: 'PWF',
    name: 'Power Failure',
    description: 'Electrical power supply disruption',
    automotive_analogue: 'Electrical system and battery failure'
})
CREATE (osf:FailureMode {
    code: 'OSF',
    name: 'Overstrain Failure',
    description: 'Mechanical overload beyond rated capacity',
    automotive_analogue: 'Drivetrain and mechanical overload'
})

// ── SOP Rule Nodes ───────────────────────────────────────────
CREATE (sop1:SOPRule {
    code: 'SOP_THERMAL',
    name: 'Thermal Stress Threshold',
    description: 'Process-air temperature delta exceeds safe operating limit',
    condition_field: 'thermal_stress',
    condition_operator: 'GREATER_THAN',
    condition_value: 8.5,
    priority: 1,
    severity: 'CRITICAL'
})
CREATE (sop2:SOPRule {
    code: 'SOP_WEAR',
    name: 'Tool Wear Criticality',
    description: 'Tool wear ratio exceeds 70 percent of rated life',
    condition_field: 'wear_criticality',
    condition_operator: 'GREATER_THAN',
    condition_value: 0.70,
    priority: 2,
    severity: 'HIGH'
})
CREATE (sop3:SOPRule {
    code: 'SOP_LOAD',
    name: 'Mechanical Load Limit',
    description: 'Torque-speed product exceeds safe mechanical boundary',
    condition_field: 'mechanical_load',
    condition_operator: 'GREATER_THAN',
    condition_value: 9000,
    priority: 3,
    severity: 'HIGH'
})
CREATE (sop4:SOPRule {
    code: 'SOP_RISK',
    name: 'Composite Failure Risk',
    description: 'Multiple failure mode signals active simultaneously',
    condition_field: 'failure_risk_score',
    condition_operator: 'GREATER_THAN',
    condition_value: 1,
    priority: 4,
    severity: 'MEDIUM'
})

// ── Mitigation Action Nodes ──────────────────────────────────
CREATE (m1:Mitigation {
    vector: 'EMERGENCY_LINE_SHUTDOWN',
    description: 'Immediately halt production line, isolate thermal source',
    estimated_cost: 5000,
    estimated_resolution_mins: 45,
    jit_impact: 'HIGH'
})
CREATE (m2:Mitigation {
    vector: 'SCHEDULED_TOOL_REPLACEMENT',
    description: 'Trigger urgent tool swap before next production cycle',
    estimated_cost: 800,
    estimated_resolution_mins: 20,
    jit_impact: 'MEDIUM'
})
CREATE (m3:Mitigation {
    vector: 'LOAD_REDISTRIBUTION',
    description: 'Reduce rotational speed and redistribute torque load',
    estimated_cost: 300,
    estimated_resolution_mins: 10,
    jit_impact: 'LOW'
})
CREATE (m4:Mitigation {
    vector: 'PREDICTIVE_MAINTENANCE_ALERT',
    description: 'Flag unit for inspection before next shift begins',
    estimated_cost: 150,
    estimated_resolution_mins: 5,
    jit_impact: 'LOW'
})

// ── Rule → Mitigation Relationships ─────────────────────────
WITH sop1, sop2, sop3, sop4, m1, m2, m3, m4, twf, hdf, pwf, osf
CREATE (sop1)-[:TRIGGERS_ACTION]->(m1)
CREATE (sop2)-[:TRIGGERS_ACTION]->(m2)
CREATE (sop3)-[:TRIGGERS_ACTION]->(m3)
CREATE (sop4)-[:TRIGGERS_ACTION]->(m4)

// ── Rule Priority Ordering ───────────────────────────────────
CREATE (sop1)-[:TAKES_PRIORITY_OVER]->(sop2)
CREATE (sop2)-[:TAKES_PRIORITY_OVER]->(sop3)
CREATE (sop3)-[:TAKES_PRIORITY_OVER]->(sop4)

// ── Failure Mode → SOP Relationships ────────────────────────
CREATE (hdf)-[:GOVERNED_BY]->(sop1)
CREATE (twf)-[:GOVERNED_BY]->(sop2)
CREATE (osf)-[:GOVERNED_BY]->(sop3)
CREATE (pwf)-[:GOVERNED_BY]->(sop4)
"""

VERIFY_QUERY = """
MATCH (s:SOPRule)-[:TRIGGERS_ACTION]->(m:Mitigation)
RETURN s.code AS rule, s.severity AS severity,
       s.condition_field AS field,
       s.condition_value AS threshold,
       m.vector AS action
ORDER BY s.priority
"""

def seed_ontology():
    conn = Neo4jConnection.get_instance()

    # Step 1 — clear existing ontology nodes
    with conn.session() as session:
        session.run(CLEAR_QUERY)
        print("[ONTOLOGY] Cleared existing ontology nodes.")

    # Step 2 — create fresh ontology
    with conn.session() as session:
        session.run(CREATE_QUERY)
        print("[ONTOLOGY] SOP rules and relationships created.")

    # Step 3 — verify
    with conn.session() as session:
        results = session.run(VERIFY_QUERY)
        records = list(results)

        if not records:
            print("[ONTOLOGY] WARNING: No rules found after seeding.")
            return

        print("\n[ONTOLOGY] Active SOP Rules in Knowledge Graph:")
        print("-" * 65)
        for r in records:
            print(f"  {r['rule']} [{r['severity']}]")
            print(f"    Condition : {r['field']} > {r['threshold']}")
            print(f"    Action    : {r['action']}")
            print()

if __name__ == "__main__":
    seed_ontology()
    Neo4jConnection.get_instance().close()