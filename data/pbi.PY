import pandas as pd
import numpy as np
import os
import sys

# Ensure we can find the data folder regardless of where script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data')

print(f"[PBI] Looking for data in: {data_dir}")

# ── Load DataCo ───────────────────────────────────────────────────
dataco_path = os.path.join(data_dir, 'DataCoSupplyChainDataset.csv')
print(f"[PBI] Loading DataCo from: {dataco_path}")

df = pd.read_csv(dataco_path, encoding='latin1')
print(f"[PBI] Loaded {len(df):,} rows")

# ── Feature Engineering ───────────────────────────────────────────
df['order_date'] = pd.to_datetime(
    df['order date (DateOrders)'],
    dayfirst=False,
    errors='coerce'
)
df['delay_days'] = (
    df['Days for shipping (real)'] -
    df['Days for shipment (scheduled)']
)
df['late_risk'] = (df['delay_days'] > 0).astype(int)
df['risk_level'] = pd.cut(
    df['delay_days'],
    bins=[-999, 0, 1, 3, 999],
    labels=['ON_TIME', 'MINOR_DELAY', 'MODERATE_DELAY', 'CRITICAL_DELAY']
)
df['part_type'] = (df['Product Category Id'] % 4).map({
    0: 'ENGINE_BLOCK',
    1: 'TRANSMISSION',
    2: 'DOOR_PANEL',
    3: 'CHASSIS_FRAME'
})
df['line_id'] = df['part_type'].map({
    'ENGINE_BLOCK': 'LINE_A',
    'TRANSMISSION': 'LINE_B',
    'DOOR_PANEL':   'LINE_C',
    'CHASSIS_FRAME':'LINE_D'
})
df['week'] = df['order_date'].dt.to_period('W').astype(str)
df['month'] = df['order_date'].dt.to_period('M').astype(str)
df['urgency_score'] = df['Shipping Mode'].map({
    'Same Day': 3,
    'First Class': 2,
    'Second Class': 1,
    'Standard Class': 0
}).fillna(0).astype(int)

# ── Export 1: Weekly Forecast ─────────────────────────────────────
weekly = df.groupby('week').agg(
    total_orders=('Order Id', 'count'),
    avg_delay=('delay_days', 'mean'),
    at_risk_orders=('late_risk', 'sum'),
    pct_late=('late_risk', 'mean'),
    critical_delays=('delay_days', lambda x: (x > 3).sum())
).reset_index()

weekly['rolling_avg_delay'] = (
    weekly['avg_delay'].rolling(4, min_periods=1).mean().round(3)
)
weekly['upper_bound'] = (weekly['rolling_avg_delay'] + 0.5).round(3)
weekly['lower_bound'] = (weekly['rolling_avg_delay'] - 0.5).round(3)
weekly['pct_late'] = (weekly['pct_late'] * 100).round(1)

out1 = os.path.join(data_dir, 'pbi_deljit_weekly_forecast.csv')
weekly.to_csv(out1, index=False)
print(f"[PBI] Weekly forecast → {out1} ({len(weekly)} weeks)")

# ── Export 2: Order-Level Detail ──────────────────────────────────
detail_cols = [
    'Order Id', 'order_date', 'delay_days', 'late_risk',
    'risk_level', 'part_type', 'line_id', 'week', 'month',
    'Shipping Mode', 'urgency_score', 'Order Item Quantity',
    'Order Region', 'Delivery Status', 'Sales',
    'Order Profit Per Order'
]
detail = df[detail_cols].copy()
detail.columns = [
    'OrderID', 'OrderDate', 'DelayDays', 'LateRisk',
    'RiskLevel', 'PartType', 'LineID', 'Week', 'Month',
    'ShippingMode', 'UrgencyScore', 'Quantity',
    'Region', 'DeliveryStatus', 'Sales', 'Profit'
]
detail['DelayDays'] = detail['DelayDays'].round(2)

out2 = os.path.join(data_dir, 'pbi_delivery_orders.csv')
detail.to_csv(out2, index=False)
print(f"[PBI] Order detail    → {out2} ({len(detail):,} rows)")

# ── Export 3: Line-Level Summary ──────────────────────────────────
line_summary = df.groupby(['line_id', 'risk_level']).agg(
    order_count=('Order Id', 'count'),
    avg_delay=('delay_days', 'mean'),
    total_sales=('Sales', 'sum')
).reset_index()
line_summary.columns = [
    'LineID', 'RiskLevel', 'OrderCount', 'AvgDelay', 'TotalSales'
]
line_summary['AvgDelay'] = line_summary['AvgDelay'].round(3)
line_summary['TotalSales'] = line_summary['TotalSales'].round(2)

out3 = os.path.join(data_dir, 'pbi_line_summary.csv')
line_summary.to_csv(out3, index=False)
print(f"[PBI] Line summary    → {out3} ({len(line_summary)} rows)")

# ── Export 4: Region Risk ─────────────────────────────────────────
region = df.groupby('Order Region').agg(
    total_orders=('Order Id', 'count'),
    at_risk=('late_risk', 'sum'),
    avg_delay=('delay_days', 'mean'),
    critical=('delay_days', lambda x: (x > 3).sum())
).reset_index()
region['pct_at_risk'] = (
    region['at_risk'] / region['total_orders'] * 100
).round(1)
region['avg_delay'] = region['avg_delay'].round(3)
region.columns = [
    'Region', 'TotalOrders', 'AtRisk', 'AvgDelay',
    'CriticalDelays', 'PctAtRisk'
]

out4 = os.path.join(data_dir, 'pbi_region_risk.csv')
region.to_csv(out4, index=False)
print(f"[PBI] Region risk     → {out4} ({len(region)} regions)")

# ── Summary ───────────────────────────────────────────────────────
print()
print("=" * 55)
print("   POWER BI EXPORT COMPLETE")
print("=" * 55)
print(f"  Weekly forecast  : pbi_deljit_weekly_forecast.csv")
print(f"  Order detail     : pbi_delivery_orders.csv")
print(f"  Line summary     : pbi_line_summary.csv")
print(f"  Region risk      : pbi_region_risk.csv")
print()
print("  Weekly forecast preview (last 4 weeks):")
print(weekly[['week','avg_delay','at_risk_orders',
              'rolling_avg_delay','upper_bound','lower_bound']].tail(4).to_string(index=False))