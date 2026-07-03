import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.neo4j_driver import Neo4jConnection


class EQAEngine:
    """
    Embodied Question Answering Engine.

    Answers plain language questions about Digital Twin
    decisions by traversing the Neo4j Knowledge Graph.

    Rationale for graph traversal over string matching:
    Every decision, rule violation, and mitigation is stored
    as a graph node with relationships. Answering "why was
    LINE_A halted?" requires traversing:
    AuditEntry → agent → rule → mitigation → cost

    This is only possible because our audit trail IS the
    graph — not a log file. Log files require regex parsing.
    Graph traversal gives structured, queryable provenance.
    """

    def __init__(self):
        self.conn = Neo4jConnection.get_instance()

    def answer(self, question: str) -> str:
        """
        Routes plain language question to correct
        graph traversal query.

        Supported question patterns:
        - "why is LINE_A at risk"
        - "what decisions were made for order 12345"
        - "which machines are critical"
        - "show compliance status"
        - "what is the buffer status"
        - "which suppliers are available for LINE_B"
        - "show audit trail"
        - "what is the total cost"
        """
        q = question.lower().strip()

        if "why" in q and "line" in q:
            line_id = self._extract_line_id(q)
            return self._why_line_at_risk(line_id)

        elif "order" in q and any(
            c.isdigit() for c in q
        ):
            order_id = self._extract_number(q)
            return self._order_decisions(order_id)

        elif "critical" in q and "machine" in q:
            return self._critical_machines()

        elif "compliance" in q or "status" in q:
            return self._compliance_summary()

        elif "buffer" in q:
            return self._buffer_status()

        elif "supplier" in q and "line" in q:
            line_id = self._extract_line_id(q)
            return self._available_suppliers(line_id)

        elif "audit" in q:
            return self._audit_trail()

        elif "cost" in q:
            return self._total_cost()

        elif "shift" in q or "worker" in q:
            return self._shift_coverage()
        
        elif "vehicle" in q or "bom" in q or "completion" in q:
            return self._vehicle_completion_risk()

        else:
            return self._general_summary()

    def _why_line_at_risk(self, line_id: str) -> str:
        """Explains why a production line is at risk."""
        results = self.conn.session().run("""
            MATCH (a:AuditEntry)
            WHERE a.compliance_status IN
                  ['VETO_RAISED', 'VIOLATION_DETECTED',
                   'ESCALATED', 'SOP_VETO', 'HARD_VETO',
                   'COMPLIANCE_BREACH']
            AND (a.decision CONTAINS $line_id
                 OR a.event_type CONTAINS $line_id)
            RETURN a.agent_name AS agent,
                   a.event_type AS event,
                   a.decision AS decision,
                   a.symbolic_rule_fired AS rule,
                   a.neural_confidence AS confidence,
                   a.cost_of_action AS cost,
                   a.timestamp AS timestamp
            ORDER BY a.timestamp DESC
            LIMIT 5
        """, line_id=line_id)

        records = list(results)

        if not records:
            # Check graph directly
            line_data = self.conn.session().run("""
                MATCH (l:ProductionLine {line_id: $line_id})
                OPTIONAL MATCH (d:DeliveryOrder)-[:DELIVERS_TO]->(l)
                WHERE d.late_risk = 1
                RETURN l.buffer_pct AS buffer_pct,
                       l.buffer_units AS buffer_units,
                       l.safety_stock_units AS safety_stock,
                       count(d) AS at_risk_orders
            """, line_id=line_id).single()

            if line_data:
                return (
                    f"\n[EQA] Risk Assessment for {line_id}:\n"
                    f"  Buffer level    : {line_data['buffer_pct']}%\n"
                    f"  Buffer units    : {line_data['buffer_units']}\n"
                    f"  Safety stock    : {line_data['safety_stock']}\n"
                    f"  At-risk orders  : {line_data['at_risk_orders']}\n"
                    f"  Agent decisions : None recorded yet"
                )
            return f"[EQA] No risk data found for {line_id}."

        response = f"\n[EQA] Why {line_id} is at risk:\n"
        response += "-" * 45 + "\n"
        for r in records:
            response += (
                f"  Agent     : {r['agent']}\n"
                f"  Event     : {r['event']}\n"
                f"  Decision  : {r['decision']}\n"
                f"  Rule      : {r['rule']}\n"
                f"  Confidence: {r['confidence']:.2%}\n"
                f"  Cost      : €{r['cost']:,.2f}\n"
                f"  Time      : {r['timestamp']}\n"
                f"  {'-'*40}\n"
            )
        return response

    def _order_decisions(self, order_id: int) -> str:
        """Shows all decisions made for a specific order."""
        result = self.conn.session().run("""
            MATCH (d:DeliveryOrder {order_id: $order_id})
            OPTIONAL MATCH (d)-[:DELIVERS_TO]->(l:ProductionLine)
            OPTIONAL MATCH (d)-[:FULFILLED_BY]->(s:Supplier)
            RETURN d.order_id AS order_id,
                   d.delivery_status AS status,
                   d.delay_days AS delay_days,
                   d.risk_level AS risk_level,
                   d.part_type AS part_type,
                   l.line_id AS line_id,
                   s.name AS supplier
        """, order_id=order_id).single()

        if not result:
            return f"[EQA] Order {order_id} not found in graph."

        return (
            f"\n[EQA] Order {order_id} Details:\n"
            f"  Status      : {result['status']}\n"
            f"  Delay       : {result['delay_days']} days\n"
            f"  Risk level  : {result['risk_level']}\n"
            f"  Part type   : {result['part_type']}\n"
            f"  Line        : {result['line_id']}\n"
            f"  Supplier    : {result['supplier']}\n"
        )

    def _critical_machines(self) -> str:
        """Lists all machines with CRITICAL risk."""
        results = self.conn.session().run("""
            MATCH (m:ProductionHour)-[:OPERATES_ON]->(l:ProductionLine)
            WHERE m.compliance_status = 'VIOLATION_DETECTED'
            AND m.triggered_sop IN ['SOP_THERMAL', 'SOP_WEAR']
            RETURN m.uid AS uid,
                   m.triggered_sop AS sop,
                   m.wear_criticality AS wear,
                   m.thermal_stress AS thermal,
                   m.mitigation_action AS action,
                   l.line_id AS line_id
            ORDER BY m.wear_criticality DESC
            LIMIT 10
        """)

        records = list(results)
        if not records:
            return "[EQA] No critical machines found."

        response = f"\n[EQA] Critical Machines ({len(records)}):\n"
        response += "-" * 45 + "\n"
        for r in records:
            response += (
                f"  UID {r['uid']} on {r['line_id']}\n"
                f"    SOP violated : {r['sop']}\n"
                f"    Wear         : {r['wear']:.1%}\n"
                f"    Thermal      : {r['thermal']:.2f}\n"
                f"    Action       : {r['action']}\n\n"
            )
        return response

    def _compliance_summary(self) -> str:
        """Overall compliance status across the Digital Twin."""
        results = self.conn.session().run("""
            MATCH (a:AuditEntry)
            RETURN a.compliance_status AS status,
                   count(a) AS count
            ORDER BY count DESC
        """)

        records = list(results)
        if not records:
            return "[EQA] No compliance records found yet."

        response = "\n[EQA] Compliance Summary:\n"
        response += "-" * 45 + "\n"
        total = sum(r["count"] for r in records)
        for r in records:
            pct = r["count"] / total * 100
            response += (
                f"  {r['status']}: "
                f"{r['count']} ({pct:.1f}%)\n"
            )
        response += f"\n  Total decisions: {total}\n"
        return response

    def _buffer_status(self) -> str:
        """Current buffer status per production line."""
        results = self.conn.session().run("""
            MATCH (l:ProductionLine)
            RETURN l.line_id AS line_id,
                   l.name AS name,
                   l.buffer_units AS buffer_units,
                   l.buffer_capacity AS capacity,
                   l.buffer_pct AS buffer_pct,
                   l.safety_stock_units AS safety_stock
            ORDER BY l.buffer_pct ASC
        """)

        records = list(results)
        if not records:
            return "[EQA] No production line data found."

        response = "\n[EQA] Buffer Status by Line:\n"
        response += "-" * 45 + "\n"
        for r in records:
            risk = (
                "🔴 CRITICAL" if r["buffer_pct"] < 15
                else "🟡 WARNING" if r["buffer_pct"] < 30
                else "🟢 OK"
            )
            response += (
                f"  {r['line_id']} — {r['name']}\n"
                f"    Buffer : {r['buffer_pct']}% "
                f"({r['buffer_units']}/{r['capacity']} units)\n"
                f"    Safety : {r['safety_stock']} units\n"
                f"    Status : {risk}\n\n"
            )
        return response

    def _available_suppliers(self, line_id: str) -> str:
        """Lists available suppliers for a production line."""
        results = self.conn.session().run("""
            MATCH (s:Supplier)-[r]->(l:ProductionLine {line_id: $line_id})
            RETURN s.name AS name,
                   s.supplier_id AS supplier_id,
                   s.supplier_tier AS tier,
                   s.reliability_score AS reliability,
                   s.lead_time_days AS lead_time,
                   s.cost_per_unit AS cost,
                   type(r) AS relationship
            ORDER BY s.reliability_score DESC
        """, line_id=line_id)

        records = list(results)
        if not records:
            return f"[EQA] No suppliers found for {line_id}."

        response = f"\n[EQA] Suppliers for {line_id}:\n"
        response += "-" * 45 + "\n"
        for r in records:
            response += (
                f"  {r['name']} [{r['tier']}]\n"
                f"    Reliability : {r['reliability']:.0%}\n"
                f"    Lead time   : {r['lead_time']} days\n"
                f"    Cost/unit   : €{r['cost']:,}\n\n"
            )
        return response

    def _audit_trail(self) -> str:
        """Recent audit entries across all agents."""
        results = self.conn.session().run("""
            MATCH (a:AuditEntry)
            RETURN a.agent_name AS agent,
                   a.event_type AS event,
                   a.decision AS decision,
                   a.symbolic_rule_fired AS rule,
                   a.neural_confidence AS confidence,
                   a.compliance_status AS status,
                   a.timestamp AS timestamp
            ORDER BY a.timestamp DESC
            LIMIT 10
        """)

        records = list(results)
        if not records:
            return "[EQA] No audit entries found."

        response = "\n[EQA] Recent Audit Trail:\n"
        response += "-" * 45 + "\n"
        for r in records:
            response += (
                f"  [{r['timestamp']}]\n"
                f"  Agent     : {r['agent']}\n"
                f"  Decision  : {r['decision']}\n"
                f"  Rule      : {r['rule']}\n"
                f"  Status    : {r['status']}\n\n"
            )
        return response

    def _total_cost(self) -> str:
        """Total mitigation cost across all decisions."""
        result = self.conn.session().run("""
            MATCH (a:AuditEntry)
            WHERE a.cost_of_action > 0
            RETURN sum(a.cost_of_action) AS total,
                   count(a) AS actions,
                   avg(a.cost_of_action) AS avg_cost,
                   max(a.cost_of_action) AS max_cost
        """).single()

        if not result or not result["total"]:
            return "[EQA] No cost data recorded yet."

        return (
            f"\n[EQA] Mitigation Cost Summary:\n"
            f"  Total cost   : €{result['total']:,.2f}\n"
            f"  Actions      : {result['actions']}\n"
            f"  Average cost : €{result['avg_cost']:,.2f}\n"
            f"  Highest cost : €{result['max_cost']:,.2f}\n"
        )

    def _shift_coverage(self) -> str:
        """Current shift worker coverage per line."""
        results = self.conn.session().run("""
            MATCH (l:ProductionLine)
            OPTIONAL MATCH (w:ShiftWorker)-[:CERTIFIED_FOR]->(l)
            WHERE w.availability = 1
            RETURN l.line_id AS line_id,
                   l.shift_requirement AS required,
                   count(w) AS available
            ORDER BY l.line_id
        """)

        records = list(results)
        if not records:
            return "[EQA] No shift data found."

        response = "\n[EQA] Shift Coverage by Line:\n"
        response += "-" * 45 + "\n"
        for r in records:
            available = r["available"]
            required = r["required"]
            status = (
                "🔴 UNDERSTAFFED"
                if available < required
                else "🟢 SUFFICIENT"
            )
            response += (
                f"  {r['line_id']}: "
                f"{available}/{required} certified workers "
                f"| {status}\n"
            )
        return response

    def _general_summary(self) -> str:
        """General Digital Twin health summary."""
        with self.conn.session() as session:
            nodes = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label,
                       count(n) AS count
                ORDER BY count DESC
            """)
            node_counts = {
                r["label"]: r["count"] for r in nodes
            }

        return (
            f"\n[EQA] NESO-DT Digital Twin Summary:\n"
            f"  Production lines  : "
            f"{node_counts.get('ProductionLine', 0)}\n"
            f"  Delivery orders   : "
            f"{node_counts.get('DeliveryOrder', 0):,}\n"
            f"  Machine units     : "
            f"{node_counts.get('ProductionHour', 0):,}\n"
            f"  Suppliers         : "
            f"{node_counts.get('Supplier', 0)}\n"
            f"  Shift workers     : "
            f"{node_counts.get('ShiftWorker', 0)}\n"
            f"  SOP rules         : "
            f"{node_counts.get('SOPRule', 0)}\n"
            f"  Regulatory rules  : "
            f"{node_counts.get('RegulatoryRule', 0)}\n"
            f"\n  Ask me:\n"
            f"  'why is LINE_A at risk'\n"
            f"  'which machines are critical'\n"
            f"  'show compliance status'\n"
            f"  'what is the buffer status'\n"
            f"  'show audit trail'\n"
            f"  'what is the total cost'\n"
        )

    def _extract_line_id(self, text: str) -> str:
        """Extracts LINE_X from question text."""
        for token in text.upper().split():
            if token.startswith("LINE_"):
                return token
        return "LINE_A"

    def _extract_number(self, text: str) -> int:
        """Extracts first number from question text."""
        for token in text.split():
            if token.isdigit():
                return int(token)
        return 0
    
    def _vehicle_completion_risk(self) -> str:
            """Shows vehicle completion risk from BOM analysis."""
            results = self.conn.session().run("""
                MATCH (v:Vehicle)-[r:AT_RISK]->(b:BOMItem)
                    -[:FULFILLED_BY]->(l:ProductionLine)
                RETURN v.model_id AS model,
                    v.name AS name,
                    v.daily_target AS daily_target,
                    b.part_type AS blocked_part,
                    l.line_id AS line_id,
                    r.severity AS severity,
                    r.wip_cost_eur AS wip_cost
                ORDER BY r.severity DESC
            """)
            records = list(results)

            if not records:
                bom = self.conn.session().run("""
                    MATCH (v:Vehicle)-[r:REQUIRES]->(b:BOMItem)
                        -[:FULFILLED_BY]->(l:ProductionLine)
                    RETURN v.model_id AS model,
                        v.daily_target AS target,
                        b.part_type AS part,
                        r.total_daily_qty AS total_qty,
                        l.line_id AS line
                    ORDER BY v.model_id, b.part_type
                """)
                bom_records = list(bom)

                if not bom_records:
                    return "[EQA] No BOM data found. Run python -m data.bom_generator first."

                response = "\n[EQA] Vehicle BOM Summary (no active risks):\n"
                response += "-" * 55 + "\n"
                current_model = None
                for r in bom_records:
                    if r["model"] != current_model:
                        current_model = r["model"]
                        response += f"\n  {r['model']} ({r['target']} vehicles/day)\n"
                    response += (f"    {r['part']:<16}: "
                                f"{r['total_qty']} units/day → {r['line']}\n")
                return response

            response = "\n[EQA] Vehicle Completion Risk:\n"
            response += "-" * 55 + "\n"
            total_blocked = 0
            total_wip = 0
            for r in records:
                response += (
                    f"  {r['model']} — {r['name']}\n"
                    f"    Blocked part : {r['blocked_part']}\n"
                    f"    Line         : {r['line_id']}\n"
                    f"    Severity     : {r['severity']}\n"
                    f"    Daily blocked: {r['daily_target']} vehicles\n"
                    f"    WIP cost     : €{r['wip_cost']:,}\n\n"
                )
                total_blocked += r["daily_target"]
                total_wip += r["wip_cost"] or 0

            response += f"  Total vehicles at risk : {total_blocked}\n"
            response += f"  Total WIP cost         : €{total_wip:,}\n"
            return response


if __name__ == "__main__":
    eqa = EQAEngine()
    print("[EQA] Engine ready. Type questions or 'exit'.")
    print("[EQA] Examples:")
    print("  why is LINE_A at risk")
    print("  which machines are critical")
    print("  show compliance status")
    print("  what is the buffer status")
    print("  show audit trail")
    print()

    while True:
        question = input("EQA > ").strip()
        if question.lower() == "exit":
            break
        if question:
            print(eqa.answer(question))