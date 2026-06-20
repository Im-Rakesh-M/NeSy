import pandas as pd

df = pd.read_csv('data/ai4i2020.csv')

df['thermal_stress'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['mechanical_load'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
df['wear_criticality'] = df['Tool wear [min]'] / 240
df['failure_risk_score'] = df['TWF'] + df['HDF'] + df['PWF'] + df['OSF'] + df['RNF']

print("=== THERMAL STRESS ===")
print(df['thermal_stress'].describe().round(2))
print("90th pct:", round(df['thermal_stress'].quantile(0.90), 2))
print("95th pct:", round(df['thermal_stress'].quantile(0.95), 2))
print("99th pct:", round(df['thermal_stress'].quantile(0.99), 2))

print("\n=== MECHANICAL LOAD ===")
print(df['mechanical_load'].describe().round(2))
print("90th pct:", round(df['mechanical_load'].quantile(0.90), 2))
print("95th pct:", round(df['mechanical_load'].quantile(0.95), 2))

print("\n=== WEAR CRITICALITY ===")
print(df['wear_criticality'].describe().round(2))
print("90th pct:", round(df['wear_criticality'].quantile(0.90), 2))
print("95th pct:", round(df['wear_criticality'].quantile(0.95), 2))

print("\n=== THERMAL STRESS AT MACHINE FAILURE ===")
print(df[df['Machine failure'] == 1]['thermal_stress'].describe().round(2))