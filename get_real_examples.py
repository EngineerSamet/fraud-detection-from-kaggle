"""
Get real examples from dataset for Streamlit app
"""
import pandas as pd
import json

# Load dataset
df = pd.read_csv('data/raw/creditcard.csv')

print(f"Total transactions: {len(df)}")
print(f"Fraud: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")
print(f"Normal: {(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.2f}%)")

# Get random examples
normal_transaction = df[df['Class']==0].sample(1, random_state=42).iloc[0]
fraud_transaction = df[df['Class']==1].sample(1, random_state=42).iloc[0]

print("\n" + "="*60)
print("REAL NORMAL TRANSACTION (Class=0)")
print("="*60)
normal_dict = normal_transaction.drop('Class').to_dict()
print(json.dumps(normal_dict, indent=2))

print("\n" + "="*60)
print("REAL FRAUD TRANSACTION (Class=1)")
print("="*60)
fraud_dict = fraud_transaction.drop('Class').to_dict()
print(json.dumps(fraud_dict, indent=2))

# Compare V value ranges
print("\n" + "="*60)
print("V VALUE COMPARISON")
print("="*60)
v_cols = [f'V{i}' for i in range(1, 29)]

print(f"\n{'Feature':<8} {'Normal':>12} {'Fraud':>12} {'Difference':>12}")
print("-" * 50)

for col in v_cols:
    normal_val = normal_dict[col]
    fraud_val = fraud_dict[col]
    diff = abs(fraud_val - normal_val)
    print(f"{col:<8} {normal_val:>12.4f} {fraud_val:>12.4f} {diff:>12.4f}")

print(f"\nAmount   {normal_dict['Amount']:>12.2f} {fraud_dict['Amount']:>12.2f}")
print(f"Time     {normal_dict['Time']:>12.0f} {fraud_dict['Time']:>12.0f}")

# Statistical comparison
print("\n" + "="*60)
print("FRAUD vs NORMAL STATISTICS")
print("="*60)

fraud_data = df[df['Class']==1][v_cols + ['Amount']]
normal_data = df[df['Class']==0][v_cols + ['Amount']]

print(f"\n{'Feature':<8} {'Normal Mean':>12} {'Fraud Mean':>12} {'Difference':>12}")
print("-" * 50)

for col in v_cols[:10]:  # First 10 V features
    normal_mean = normal_data[col].mean()
    fraud_mean = fraud_data[col].mean()
    diff = abs(fraud_mean - normal_mean)
    print(f"{col:<8} {normal_mean:>12.4f} {fraud_mean:>12.4f} {diff:>12.4f}")

print(f"\nAmount   {normal_data['Amount'].mean():>12.2f} {fraud_data['Amount'].mean():>12.2f}")
