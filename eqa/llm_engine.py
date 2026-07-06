import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import anthropic
from core.neo4j_driver import Neo4jConnection

# ── Graph Schema Context ───────────────────────────────────────────
# This tells the LLM exactly what nodes and relationships exist
# so it generates valid Cypher — not hallucinated queries.

GRAPH_SCHEMA = """
You are a Neo4j Cypher expert for NESO-DT — a Digital Twin for automotive JIT manufacturing.

NODE TYPES AND KEY PROPERTIES:
- ProductionLine {line_id, name, part_type, buffer_units, buffer_capacity, buffer_pct, safety_stock_units, cycle_time_mins, shift_requirement, certified_role, buffer_warning_pct, buffer_critical_pct, jit_delay_threshold_mins}
- DeliveryOrder {order_id, delay_days, delay_mins, late_risk, risk_level, part_type, line_id, shipping_mode, region, quantity, on_time, delivery_status}
- ProductionHour {uid, thermal_stress, mechanical_load, wear_criticality, failure_risk_score, machine_failure, compliance_status, triggered_sop, mitigation_action, mitigation_cost}
- Supplier {supplier_id, name, part_type, supplier_tier, reliability_score, lead_time_days, cost_per_unit, status}
- ShiftWorker {worker_id, name, shift_id, shift_start, shift_end, primary_role, availability, overtime_eligible, experience_years}
- SOPRule {code, name, condition_field, condition_operator, condition_value, severity, priority}
- Mitigation {vector, description, estimated_cost, estimated_resolution_mins, jit_impact}
- RegulatoryRule {rule_id, name, description, jurisdiction, penalty_eur, hard_constraint, applies_to}
- AuditEntry {agent_name, event_type, decision, symbolic_rule_fired, neural_confidence, compliance_status, cost_of_action, timestamp, alternative_considered}
- Vehicle {model_id, name, daily_target, takt_time_mins, build_date, completion_risk}
- BOMItem {part_type, part_number, daily_requirement, line_id, criticality, unit_cost_eur}
- BuildSchedule {schedule_id, build_date, total_vehicles, status}
- AgentBelief {agent, key, value, updated}

RELATIONSHIP TYPES:
- (ProductionHour)-[:OPERATES_ON]->(ProductionLine)
- (DeliveryOrder)-[:DELIVERS_TO]->(ProductionLine)
- (DeliveryOrder)-[:FULFILLED_BY]->(Supplier)
- (ProductionHour)-[:VIOLATED_CONSTRAINT]->(SOPRule)
- (ProductionHour)-[:TRIGGERED_MITIGATION]->(Mitigation)
- (SOPRule)-[:TRIGGERS_ACTION]->(Mitigation)
- (SOPRule)-[:TAKES_PRIORITY_OVER]->(SOPRule)
- (Supplier)-[:PRIMARY_SUPPLIER_FOR]->(ProductionLine)
- (Supplier)-[:BACKUP_SUPPLIER_FOR]->(ProductionLine)
- (ShiftWorker)-[:CERTIFIED_FOR]->(ProductionLine)
- (ProductionLine)-[:GOVERNED_BY]->(SOPRule)
- (ProductionLine)-[:MUST_COMPLY_WITH]->(RegulatoryRule)
- (Vehicle)-[:REQUIRES {quantity_per_vehicle, total_daily_qty, criticality}]->(BOMItem)
- (BOMItem)-[:FULFILLED_BY]->(ProductionLine)
- (Vehicle)-[:AT_RISK {severity, wip_cost_eur, neural_confidence}]->(BOMItem)
- (Vehicle)-[:SCHEDULED_FOR]->(BuildSchedule)
- (BOMItem)-[:SOURCED_FROM {tier}]->(Supplier)

IMPORTANT CYPHER RULES:
1. Always use LIMIT to prevent huge result sets (max 20 for tables, 25 for graphs)
2. Use OPTIONAL MATCH for relationships that may not exist
3. Property names are case sensitive — use exact names above
4. For counting use count(n) not COUNT(n)
5. For cost formatting use round(x, 2)
6. Always ORDER BY something meaningful
"""

ANSWER_PROMPT = """
You are a shop floor intelligence assistant for NESO-DT — an automotive JIT Digital Twin.

The user asked: {question}

The Cypher query that was run:
{cypher}

The raw results from Neo4j:
{results}

Convert these results into a clear, concise plain English answer.
Be specific with numbers. Use EUR for costs. Mention line IDs, agent names, and rule codes where relevant.
If results are empty, explain what that means in context.
Keep the answer under 150 words.
"""


class LLMEQAEngine:
    """
    LLM-powered Embodied Question Answering Engine.

    Unlike the rule-based EQA which routes keywords to fixed
    Cypher queries, this engine:
    1. Sends the question + graph schema to Claude
    2. Claude generates the appropriate Cypher query
    3. System runs the query against Neo4j
    4. Claude converts raw results to plain English

    This allows ANY natural language question about the
    Digital Twin — not just the 9 predefined patterns.
    """

    def __init__(self):
        self.conn = Neo4jConnection.get_instance()
        self.client = anthropic.Anthropic()
        self._query_history = []

    def _generate_cypher(self, question: str) -> str:
        """
        Asks Claude to generate a Cypher query for the question.
        Returns only the Cypher query string.
        """
        prompt = f"""{GRAPH_SCHEMA}

The user asked this question about the Digital Twin:
"{question}"

Generate a single Cypher query to answer this question.
Return ONLY the Cypher query — no explanation, no markdown, no backticks.
The query must be valid Neo4j Cypher that will run without errors."""

        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        cypher = message.content[0].text.strip()

        # Clean up any markdown formatting the model might add
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        return cypher

    def _run_cypher(self, cypher: str) -> list:
        """Runs the generated Cypher against Neo4j."""
        try:
            with self.conn.session() as session:
                result = session.run(cypher)
                records = [dict(r) for r in result]
                return records
        except Exception as e:
            return [{"error": str(e)}]

    def _generate_answer(
        self, question: str, cypher: str, results: list
    ) -> str:
        """
        Asks Claude to convert raw Neo4j results into
        plain English answer.
        """
        prompt = ANSWER_PROMPT.format(
            question=question,
            cypher=cypher,
            results=str(results[:10])  # Limit to first 10 results
        )

        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text.strip()

    def answer(self, question: str) -> str:
        """
        Full pipeline: question → Cypher → Neo4j → plain English.
        """
        print(f"\n[LLM EQA] Question: {question}")
        print("[LLM EQA] Generating Cypher query...")

        # Step 1: Generate Cypher
        cypher = self._generate_cypher(question)
        print(f"[LLM EQA] Generated query:\n  {cypher[:100]}...")

        # Step 2: Run against Neo4j
        print("[LLM EQA] Running against Knowledge Graph...")
        results = self._run_cypher(cypher)

        if results and "error" in results[0]:
            # If Cypher failed, return the error clearly
            return (f"[LLM EQA] Query failed: {results[0]['error']}\n"
                    f"Generated query was: {cypher}")

        print(f"[LLM EQA] Got {len(results)} results.")

        # Step 3: Convert to plain English
        print("[LLM EQA] Generating natural language answer...")
        answer = self._generate_answer(question, cypher, results)

        # Log to history
        self._query_history.append({
            "question": question,
            "cypher": cypher,
            "result_count": len(results)
        })

        return f"\n[NESO-DT] {answer}"

    def get_query_history(self) -> list:
        """Returns history of questions and generated queries."""
        return self._query_history


if __name__ == "__main__":
    print("=" * 60)
    print("   NESO-DT LLM-POWERED EQA ENGINE")
    print("=" * 60)
    print("Ask any question about your Digital Twin.")
    print("Type 'history' to see generated queries.")
    print("Type 'exit' to quit.")
    print()

    engine = LLMEQAEngine()

    while True:
        question = input("Ask > ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        if question.lower() == "history":
            history = engine.get_query_history()
            for i, h in enumerate(history, 1):
                print(f"\n{i}. Q: {h['question']}")
                print(f"   Cypher: {h['cypher'][:80]}...")
                print(f"   Results: {h['result_count']} records")
            continue

        print(engine.answer(question))