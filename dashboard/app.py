import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from core.neo4j_driver import Neo4jConnection
from eqa.query_engine import EQAEngine

app = FastAPI(title="NESO-DT Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

conn = Neo4jConnection.get_instance()
eqa = EQAEngine()


# ── API Routes ───────────────────────────────────────────────

@app.get("/api/summary")
def get_summary():
    """Digital Twin overall health summary."""
    with conn.session() as session:
        nodes = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        node_counts = {r["label"]: r["count"] for r in nodes}

    with conn.session() as session:
        violations = session.run("""
            MATCH (p:ProductionHour)
            WHERE p.compliance_status = 'VIOLATION_DETECTED'
            RETURN count(p) AS violations
        """).single()

    with conn.session() as session:
        at_risk = session.run("""
            MATCH (d:DeliveryOrder)
            WHERE d.late_risk = 1
            RETURN count(d) AS at_risk
        """).single()

    return {
        "total_nodes": sum(node_counts.values()),
        "production_lines": node_counts.get("ProductionLine", 0),
        "delivery_orders": node_counts.get("DeliveryOrder", 0),
        "machine_units": node_counts.get("ProductionHour", 0),
        "suppliers": node_counts.get("Supplier", 0),
        "shift_workers": node_counts.get("ShiftWorker", 0),
        "sop_violations": violations["violations"] if violations else 0,
        "at_risk_deliveries": at_risk["at_risk"] if at_risk else 0
    }


@app.get("/api/lines")
def get_lines():
    """Production line status with buffer levels."""
    with conn.session() as session:
        results = session.run("""
            MATCH (l:ProductionLine)
            OPTIONAL MATCH (d:DeliveryOrder)-[:DELIVERS_TO]->(l)
            WHERE d.late_risk = 1
            OPTIONAL MATCH (w:ShiftWorker)-[:CERTIFIED_FOR]->(l)
            WHERE w.availability = 1
            RETURN l.line_id AS line_id,
                   l.name AS name,
                   l.buffer_pct AS buffer_pct,
                   l.buffer_units AS buffer_units,
                   l.safety_stock_units AS safety_stock,
                   l.part_type AS part_type,
                   l.shift_requirement AS shift_requirement,
                   count(DISTINCT d) AS at_risk_orders,
                   count(DISTINCT w) AS available_workers
            ORDER BY l.line_id
        """)
        return [dict(r) for r in results]


@app.get("/api/deliveries")
def get_deliveries(limit: int = 50):
    """Recent high-risk delivery orders."""
    with conn.session() as session:
        results = session.run("""
            MATCH (d:DeliveryOrder)-[:DELIVERS_TO]->(l:ProductionLine)
            WHERE d.late_risk = 1
            RETURN d.order_id AS order_id,
                   d.delay_days AS delay_days,
                   d.risk_level AS risk_level,
                   d.part_type AS part_type,
                   d.shipping_mode AS shipping_mode,
                   d.region AS region,
                   l.line_id AS line_id
            ORDER BY d.delay_days DESC
            LIMIT $limit
        """, limit=limit)
        return [dict(r) for r in results]


@app.get("/api/machines")
def get_machines(limit: int = 50):
    """Machine units with SOP violations."""
    with conn.session() as session:
        results = session.run("""
            MATCH (m:ProductionHour)-[:OPERATES_ON]->(l:ProductionLine)
            WHERE m.compliance_status = 'VIOLATION_DETECTED'
            RETURN m.uid AS uid,
                   m.triggered_sop AS sop,
                   m.wear_criticality AS wear_criticality,
                   m.thermal_stress AS thermal_stress,
                   m.mitigation_action AS action,
                   m.mitigation_cost AS cost,
                   l.line_id AS line_id
            ORDER BY m.wear_criticality DESC
            LIMIT $limit
        """, limit=limit)
        return [dict(r) for r in results]


@app.get("/api/suppliers/{line_id}")
def get_suppliers(line_id: str):
    """Suppliers for a specific production line."""
    with conn.session() as session:
        results = session.run("""
            MATCH (s:Supplier)-[r]->(l:ProductionLine {line_id: $line_id})
            RETURN s.supplier_id AS supplier_id,
                   s.name AS name,
                   s.supplier_tier AS tier,
                   s.reliability_score AS reliability,
                   s.lead_time_days AS lead_time,
                   s.cost_per_unit AS cost,
                   s.status AS status
            ORDER BY s.reliability_score DESC
        """, line_id=line_id)
        return [dict(r) for r in results]


@app.get("/api/audit")
def get_audit(limit: int = 20):
    """Recent audit trail entries."""
    with conn.session() as session:
        results = session.run("""
            MATCH (a:AuditEntry)
            RETURN a.agent_name AS agent,
                   a.event_type AS event,
                   a.decision AS decision,
                   a.symbolic_rule_fired AS rule,
                   a.neural_confidence AS confidence,
                   a.compliance_status AS status,
                   a.cost_of_action AS cost,
                   a.timestamp AS timestamp
            ORDER BY a.timestamp DESC
            LIMIT $limit
        """, limit=limit)
        return [dict(r) for r in results]


@app.get("/api/compliance")
def get_compliance():
    """Compliance status summary."""
    with conn.session() as session:
        results = session.run("""
            MATCH (a:AuditEntry)
            RETURN a.compliance_status AS status,
                   count(a) AS count
            ORDER BY count DESC
        """)
        return [dict(r) for r in results]


@app.get("/api/sop-rules")
def get_sop_rules():
    """All SOP rules from Knowledge Graph."""
    with conn.session() as session:
        results = session.run("""
            MATCH (s:SOPRule)-[:TRIGGERS_ACTION]->(m:Mitigation)
            RETURN s.code AS code,
                   s.name AS name,
                   s.condition_field AS field,
                   s.condition_value AS threshold,
                   s.severity AS severity,
                   m.vector AS action,
                   m.estimated_cost AS cost
            ORDER BY s.priority
        """)
        return [dict(r) for r in results]


class EQAQuery(BaseModel):
    question: str


@app.post("/api/eqa")
def ask_eqa(query: EQAQuery):
    """EQA endpoint — answer plain language questions."""
    answer = eqa.answer(query.question)
    return {"question": query.question, "answer": answer}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serves the React dashboard."""
    return HTMLResponse(content=open(
        "dashboard/index.html",
        encoding="utf-8"
    ).read())

from eqa.llm_engine import LLMEQAEngine

llm_eqa = LLMEQAEngine()

@app.post("/api/llm-eqa")
def ask_llm_eqa(query: EQAQuery):
    """LLM-powered EQA — answers any natural language question."""
    answer = llm_eqa.answer(query.question)
    return {"question": query.question, "answer": answer}