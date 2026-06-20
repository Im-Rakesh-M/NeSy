import pandas as pd
import numpy as np
import time
from core.neo4j_driver import Neo4jConnection

class AI4IDataEngine:
    """
    Loads AI4I dataset, engineers JIT supply chain features,
    and ingests production hour nodes into the Knowledge Graph.

    Feature Engineering Rationale:
    - thermal_stress    : delta between process and air temp.
                          High delta = thermal runaway risk.
    - mechanical_load   : torque x rotational speed.
                          Proxy for physical strain on machine.
    - wear_criticality  : tool wear normalized to max threshold (240 min).
                          Above 0.70 = entering danger zone.
    - failure_risk_score: sum of all active failure mode flags.
                          >1 means multiple systems failing simultaneously.
    - jit_supply_pressure: rolling 5-hour mean of failure_risk_score.
                          Captures sustained pressure on supply chain.
    """

    MAX_TOOL_WEAR = 240  # minutes — domain threshold from AI4I documentation

    def __init__(self, filepath="data/ai4i2020.csv"):
        self.filepath = filepath
        self.conn = Neo4jConnection.get_instance()

    def load_and_engineer(self):
        print("[DATA ENGINE] Loading AI4I dataset...")
        df = pd.read_csv(self.filepath)
        print(f"  -> Raw shape: {df.shape}")

        # ── Feature Engineering ──────────────────────────────
        df['thermal_stress'] = (
            df['Process temperature [K]'] - df['Air temperature [K]']
        ).round(3)

        df['mechanical_load'] = (
            df['Torque [Nm]'] * df['Rotational speed [rpm]']
        ).round(2)

        df['wear_criticality'] = (
            df['Tool wear [min]'] / self.MAX_TOOL_WEAR
        ).round(4)

        df['failure_risk_score'] = (
            df['TWF'] + df['HDF'] + df['PWF'] + df['OSF'] + df['RNF']
        )

        df['jit_supply_pressure'] = (
            df['failure_risk_score']
            .rolling(window=5, min_periods=1)
            .mean()
            .round(4)
        )

        # Encode product type as integer
        type_map = {'L': 0, 'M': 1, 'H': 2}
        df['product_type_encoded'] = df['Type'].map(type_map)

        print(f"  -> Engineered features added.")
        print(f"  -> Machine failures : {df['Machine failure'].sum()} "
              f"({df['Machine failure'].mean()*100:.1f}%)")
        print(f"  -> Thermal stress   : mean={df['thermal_stress'].mean():.2f}, "
              f"max={df['thermal_stress'].max():.2f}")
        print(f"  -> High wear units  : "
              f"{(df['wear_criticality'] > 0.70).sum()}")

        self.df = df
        return df

    def push_to_knowledge_graph(self, batch_size=500):
        """
        Pushes each production hour as a node into Neo4j.
        
        Rationale for batch_size=500:
        Aura free tier has connection limits. 500-row batches
        balance throughput against connection stability.
        """
        print(f"\n[KG] Pushing {len(self.df)} production nodes to Neo4j...")
        
        # Clear existing production nodes only
        with self.conn.session() as session:
            session.run("MATCH (p:ProductionHour) DETACH DELETE p")
            print("[KG] Cleared existing ProductionHour nodes.")

        # Build payload
        records = []
        for _, row in self.df.iterrows():
            records.append({
                "uid": int(row['UDI']),
                "product_id": str(row['Product ID']),
                "product_type": str(row['Type']),
                "product_type_encoded": int(row['product_type_encoded']),
                "air_temp": float(row['Air temperature [K]']),
                "process_temp": float(row['Process temperature [K]']),
                "rotational_speed": int(row['Rotational speed [rpm]']),
                "torque": float(row['Torque [Nm]']),
                "tool_wear": int(row['Tool wear [min]']),
                "machine_failure": int(row['Machine failure']),
                "twf": int(row['TWF']),
                "hdf": int(row['HDF']),
                "pwf": int(row['PWF']),
                "osf": int(row['OSF']),
                "rnf": int(row['RNF']),
                "thermal_stress": float(row['thermal_stress']),
                "mechanical_load": float(row['mechanical_load']),
                "wear_criticality": float(row['wear_criticality']),
                "failure_risk_score": int(row['failure_risk_score']),
                "jit_supply_pressure": float(row['jit_supply_pressure'])
            })

        # Batch ingest
        cypher = """
        UNWIND $batch AS row
        CREATE (p:ProductionHour {
            uid                 : row.uid,
            product_id          : row.product_id,
            product_type        : row.product_type,
            product_type_encoded: row.product_type_encoded,
            air_temp            : row.air_temp,
            process_temp        : row.process_temp,
            rotational_speed    : row.rotational_speed,
            torque              : row.torque,
            tool_wear           : row.tool_wear,
            machine_failure     : row.machine_failure,
            twf                 : row.twf,
            hdf                 : row.hdf,
            pwf                 : row.pwf,
            osf                 : row.osf,
            rnf                 : row.rnf,
            thermal_stress      : row.thermal_stress,
            mechanical_load     : row.mechanical_load,
            wear_criticality    : row.wear_criticality,
            failure_risk_score  : row.failure_risk_score,
            jit_supply_pressure : row.jit_supply_pressure
        })
        """

        start = time.time()
        total = len(records)

        for offset in range(0, total, batch_size):
            chunk = records[offset:offset + batch_size]
            with self.conn.session() as session:
                session.run(cypher, batch=chunk)
            print(f"  -> Committed [{offset} to "
                  f"{min(offset+batch_size, total)}]")

        elapsed = time.time() - start
        print(f"\n[KG] Ingestion complete.")
        print(f"  -> Total nodes pushed : {total}")
        print(f"  -> Time taken         : {elapsed:.2f}s")

    def evaluate_sop_rules_in_graph(self):
        """
        Reads SOP rules FROM Neo4j and evaluates them against
        production nodes entirely inside the graph.
        
        Rationale: Split into one query per rule, ordered by
        priority. First rule that matches wins — higher priority
        rules take precedence. This mirrors real factory SOP
        escalation logic.
        """
        print("\n[SOP ENGINE] Evaluating SOP rules against production nodes...")

        # Fetch rules from graph — Python never hardcodes them
        with self.conn.session() as session:
            rules = session.run("""
                MATCH (s:SOPRule)-[:TRIGGERS_ACTION]->(m:Mitigation)
                RETURN s.code AS code,
                    s.condition_field AS field,
                    s.condition_operator AS operator,
                    s.condition_value AS threshold,
                    s.severity AS severity,
                    s.priority AS priority,
                    m.vector AS action,
                    m.estimated_cost AS cost
                ORDER BY s.priority ASC
            """)
            rule_list = [dict(r) for r in rules]

        print(f"  -> Loaded {len(rule_list)} rules from graph.")

        # Evaluate each rule in priority order
        # Nodes already tagged by a higher priority rule are skipped
        for rule in rule_list:
            cypher = f"""
                MATCH (p:ProductionHour)
                WHERE p.triggered_sop IS NULL
                AND p.{rule['field']} > $threshold
                MATCH (s:SOPRule {{code: $code}})
                MATCH (s)-[:TRIGGERS_ACTION]->(m:Mitigation)
                CREATE (p)-[:VIOLATED_CONSTRAINT {{
                    evaluated_at: datetime(),
                    severity: $severity
                }}]->(s)
                CREATE (p)-[:TRIGGERED_MITIGATION {{
                    evaluated_at: datetime(),
                    cost: $cost
                }}]->(m)
                SET p.triggered_sop = $code,
                    p.mitigation_action = $action,
                    p.mitigation_cost = $cost,
                    p.compliance_status = 'VIOLATION_DETECTED'
            """
            with self.conn.session() as session:
                session.run(cypher,
                    threshold=rule['threshold'],
                    code=rule['code'],
                    severity=rule['severity'],
                    cost=rule['cost'],
                    action=rule['action']
                )
            print(f"  -> Evaluated {rule['code']} [{rule['severity']}]")

        # Mark all remaining nodes as compliant
        with self.conn.session() as session:
            session.run("""
                MATCH (p:ProductionHour)
                WHERE p.triggered_sop IS NULL
                SET p.compliance_status = 'COMPLIANT',
                    p.mitigation_action = 'NONE',
                    p.mitigation_cost = 0
            """)

        # Print violation summary
        with self.conn.session() as session:
            results = session.run("""
                MATCH (p:ProductionHour)-[:VIOLATED_CONSTRAINT]->(s:SOPRule)
                RETURN s.code AS rule,
                    s.severity AS severity,
                    count(p) AS violations
                ORDER BY violations DESC
            """)
            print("\n[SOP ENGINE] Violation Summary:")
            print("-" * 45)
            total_violations = 0
            for r in results:
                print(f"  {r['rule']} [{r['severity']}]: "
                    f"{r['violations']} violations")
                total_violations += r['violations']
            print(f"\n  Total violations : {total_violations}")
            print(f"  Compliant nodes  : {10000 - total_violations}")

if __name__ == "__main__":
    engine = AI4IDataEngine()
    engine.load_and_engineer()
    engine.push_to_knowledge_graph()
    engine.evaluate_sop_rules_in_graph()
    Neo4jConnection.get_instance().close()