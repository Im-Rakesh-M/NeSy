import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from core.neo4j_driver import Neo4jConnection

class DataCoLoader:
    """
    Loads DataCo supply chain dataset and maps delivery
    orders to the existing Digital Twin graph.

    Feature Engineering Rationale:
    - delay_days     : real - scheduled shipping days.
                       Negative = early, positive = late.
    - delay_mins     : delay_days * 1440.
                       Enables our 10-minute JIT threshold logic.
    - urgency_score  : Shipping mode encoded by speed.
                       Same Day=3, First=2, Second=1, Standard=0.
    - on_time        : 1 if delay_days <= 0, else 0.
    - part_type      : mapped from DataCo product categories
                       to our 4 OEM part types.
    - line_id        : derived from part_type → ProductionLine.
    - supplier_id    : primary supplier for that part type.
    """

    SHIPPING_URGENCY = {
        'Same Day': 3,
        'First Class': 2,
        'Second Class': 1,
        'Standard Class': 0
    }

    # Map DataCo product categories to OEM part types
    # Rationale: DataCo has 40+ product categories.
    # We map them to 4 automotive part types by category ID
    # modulo 4 — ensuring even distribution across lines.
    PART_TYPE_MAP = {
        0: 'ENGINE_BLOCK',
        1: 'TRANSMISSION',
        2: 'DOOR_PANEL',
        3: 'CHASSIS_FRAME'
    }

    LINE_MAP = {
        'ENGINE_BLOCK': 'LINE_A',
        'TRANSMISSION': 'LINE_B',
        'DOOR_PANEL': 'LINE_C',
        'CHASSIS_FRAME': 'LINE_D'
    }

    PRIMARY_SUPPLIER_MAP = {
        'ENGINE_BLOCK': 'SUP_001',
        'TRANSMISSION': 'SUP_004',
        'DOOR_PANEL': 'SUP_007',
        'CHASSIS_FRAME': 'SUP_010'
    }

    def __init__(self, filepath="data/DataCoSupplyChainDataset.csv"):
        self.filepath = filepath
        self.conn = Neo4jConnection.get_instance()

    def load_and_engineer(self):
        print("[DATACO] Loading DataCo supply chain dataset...")
        df = pd.read_csv(self.filepath, encoding='latin1')
        print(f"  -> Raw shape: {df.shape}")

        # Keep only relevant columns
        cols = [
            'Order Id',
            'order date (DateOrders)',
            'shipping date (DateOrders)',
            'Days for shipping (real)',
            'Days for shipment (scheduled)',
            'Late_delivery_risk',
            'Delivery Status',
            'Shipping Mode',
            'Order Item Quantity',
            'Order Region',
            'Product Name',
            'Product Category Id',
            'Order Status',
            'Product Status',
            'Sales',
            'Order Profit Per Order'
        ]
        df = df[cols].copy()

        # ── Feature Engineering ──────────────────────────
        df['delay_days'] = (
            df['Days for shipping (real)'] -
            df['Days for shipment (scheduled)']
        ).round(2)

        df['delay_mins'] = (df['delay_days'] * 1440).round(0)

        df['urgency_score'] = (
            df['Shipping Mode'].map(self.SHIPPING_URGENCY).fillna(0)
        )

        df['on_time'] = (df['delay_days'] <= 0).astype(int)

        df['part_type'] = (
            df['Product Category Id'] % 4
        ).map(self.PART_TYPE_MAP)

        df['line_id'] = df['part_type'].map(self.LINE_MAP)

        df['supplier_id'] = df['part_type'].map(
            self.PRIMARY_SUPPLIER_MAP
        )

        # Delivery risk level
        df['risk_level'] = pd.cut(
            df['delay_days'],
            bins=[-999, 0, 1, 3, 999],
            labels=['ON_TIME', 'MINOR_DELAY',
                    'MODERATE_DELAY', 'CRITICAL_DELAY']
        )

        # Clean nulls
        df = df.dropna(subset=['delay_days', 'part_type'])
        df['Order Id'] = df['Order Id'].astype(int)

        print(f"  -> Processed shape: {df.shape}")
        print(f"  -> On-time deliveries : "
              f"{df['on_time'].sum():,} "
              f"({df['on_time'].mean()*100:.1f}%)")
        print(f"  -> Late deliveries    : "
              f"{(df['on_time']==0).sum():,} "
              f"({(df['on_time']==0).mean()*100:.1f}%)")
        print(f"  -> Critical delays    : "
              f"{(df['risk_level']=='CRITICAL_DELAY').sum():,}")
        print(f"  -> Avg delay days     : "
              f"{df['delay_days'].mean():.2f}")

        self.df = df
        return df

    def push_to_knowledge_graph(self, batch_size=500):
        """
        Pushes DataCo delivery orders as DeliveryOrder nodes
        and links them to ProductionLine and Supplier nodes.
        """
        print(f"\n[DATACO] Pushing delivery orders to Neo4j...")

        # Clear existing delivery nodes
        with self.conn.session() as session:
            session.run(
                "MATCH (n:DeliveryOrder) DETACH DELETE n"
            )
            print("[DATACO] Cleared existing DeliveryOrder nodes.")

        records = []
        for _, row in self.df.iterrows():
            records.append({
                "order_id": int(row['Order Id']),
                "order_date": str(row['order date (DateOrders)']),
                "ship_date": str(row['shipping date (DateOrders)']),
                "days_real": float(row['Days for shipping (real)']),
                "days_scheduled": float(
                    row['Days for shipment (scheduled)']
                ),
                "delay_days": float(row['delay_days']),
                "delay_mins": float(row['delay_mins']),
                "late_risk": int(row['Late_delivery_risk']),
                "delivery_status": str(row['Delivery Status']),
                "shipping_mode": str(row['Shipping Mode']),
                "urgency_score": int(row['urgency_score']),
                "on_time": int(row['on_time']),
                "quantity": int(row['Order Item Quantity']),
                "region": str(row['Order Region']),
                "product_name": str(row['Product Name']),
                "part_type": str(row['part_type']),
                "line_id": str(row['line_id']),
                "supplier_id": str(row['supplier_id']),
                "order_status": str(row['Order Status']),
                "risk_level": str(row['risk_level']),
                "sales": float(row['Sales']),
                "profit": float(row['Order Profit Per Order'])
            })

        # Batch ingest
        cypher = """
        UNWIND $batch AS row
        CREATE (d:DeliveryOrder {
            order_id      : row.order_id,
            order_date    : row.order_date,
            ship_date     : row.ship_date,
            days_real     : row.days_real,
            days_scheduled: row.days_scheduled,
            delay_days    : row.delay_days,
            delay_mins    : row.delay_mins,
            late_risk     : row.late_risk,
            delivery_status: row.delivery_status,
            shipping_mode : row.shipping_mode,
            urgency_score : row.urgency_score,
            on_time       : row.on_time,
            quantity      : row.quantity,
            region        : row.region,
            product_name  : row.product_name,
            part_type     : row.part_type,
            line_id       : row.line_id,
            supplier_id   : row.supplier_id,
            order_status  : row.order_status,
            risk_level    : row.risk_level,
            sales         : row.sales,
            profit        : row.profit
        })
        """

        import time
        start = time.time()
        total = len(records)

        for offset in range(0, total, batch_size):
            chunk = records[offset:offset + batch_size]
            with self.conn.session() as session:
                session.run(cypher, batch=chunk)
            print(f"  -> Committed [{offset} to "
                  f"{min(offset+batch_size, total)}]")

        elapsed = time.time() - start
        print(f"\n[DATACO] Ingestion complete.")
        print(f"  -> Total orders pushed : {total:,}")
        print(f"  -> Time taken          : {elapsed:.2f}s")

    def link_to_graph(self):
        """
        Links DeliveryOrder nodes to ProductionLine
        and Supplier nodes via relationships.
        """
        print("\n[DATACO] Linking delivery orders to graph...")

        with self.conn.session() as session:
            # DeliveryOrder → ProductionLine
            session.run("""
                MATCH (d:DeliveryOrder), (l:ProductionLine)
                WHERE d.line_id = l.line_id
                CREATE (d)-[:DELIVERS_TO]->(l)
            """)
            print("  -> DeliveryOrder → ProductionLine linked.")

            # DeliveryOrder → Supplier
            session.run("""
                MATCH (d:DeliveryOrder), (s:Supplier)
                WHERE d.supplier_id = s.supplier_id
                CREATE (d)-[:FULFILLED_BY]->(s)
            """)
            print("  -> DeliveryOrder → Supplier linked.")

        # Verify
        with self.conn.session() as session:
            result = session.run("""
                MATCH (d:DeliveryOrder)-[:DELIVERS_TO]->(l:ProductionLine)
                RETURN l.line_id AS line,
                       count(d) AS orders,
                       avg(d.delay_days) AS avg_delay,
                       sum(CASE WHEN d.late_risk = 1
                           THEN 1 ELSE 0 END) AS at_risk
                ORDER BY l.line_id
            """)
            print("\n[DATACO] Delivery Distribution by Line:")
            print("-" * 55)
            for r in result:
                print(f"  {r['line']}: {r['orders']:,} orders | "
                      f"avg delay: {r['avg_delay']:.2f} days | "
                      f"at risk: {r['at_risk']:,}")

if __name__ == "__main__":
    loader = DataCoLoader()
    loader.load_and_engineer()
    loader.push_to_knowledge_graph()
    loader.link_to_graph()
    Neo4jConnection.get_instance().close()