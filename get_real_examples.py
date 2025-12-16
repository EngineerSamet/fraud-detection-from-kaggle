"""
Get real examples from dataset for Streamlit app
Extract multiple fraud and normal examples to test different patterns
"""
import pandas as pd
import json
import numpy as np

# Load dataset
df = pd.read_csv('data/raw/creditcard.csv')

print(f"Total transactions: {len(df)}")
print(f"Fraud: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")
print(f"Normal: {(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.2f}%)")

# Get multiple examples (10 fraud + 10 normal)
print("\n" + "="*80)
print("EXTRACTING 10 FRAUD + 10 NORMAL EXAMPLES")
print("="*80)

fraud_examples = []
normal_examples = []

# Get 10 fraud examples with different random states
for i, seed in enumerate([42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555], 1):
    fraud = df[df['Class']==1].sample(1, random_state=seed).iloc[0]
    fraud_dict = fraud.drop('Class').to_dict()
    fraud_examples.append({
        'id': f'fraud_{i}',
        'seed': seed,
        'data': fraud_dict
    })
    
    normal = df[df['Class']==0].sample(1, random_state=seed).iloc[0]
    normal_dict = normal.drop('Class').to_dict()
    normal_examples.append({
        'id': f'normal_{i}',
        'seed': seed,
        'data': normal_dict
    })


# Print all examples
print("\n" + "="*80)
print("FRAUD EXAMPLES")
print("="*80)
for ex in fraud_examples:
    print(f"\n--- {ex['id'].upper()} (seed={ex['seed']}) ---")
    print(f"Amount: ${ex['data']['Amount']:.2f}")
    print(f"V14={ex['data']['V14']:.2f}, V4={ex['data']['V4']:.2f}, V12={ex['data']['V12']:.2f} (Top SHAP features)")

print("\n" + "="*80)
print("NORMAL EXAMPLES")
print("="*80)
for ex in normal_examples:
    print(f"\n--- {ex['id'].upper()} (seed={ex['seed']}) ---")
    print(f"Amount: ${ex['data']['Amount']:.2f}")
    print(f"V14={ex['data']['V14']:.2f}, V4={ex['data']['V4']:.2f}, V12={ex['data']['V12']:.2f} (Top SHAP features)")


# Detailed analysis: Compare V value ranges
print("\n" + "="*80)
print("V VALUE RANGE ANALYSIS")
print("="*80)

v_cols = [f'V{i}' for i in range(1, 29)]

print(f"\n{'Feature':<8} {'Normal Range':>20} {'Fraud Range':>20}")
print("-" * 60)

for col in v_cols:
    normal_vals = [ex['data'][col] for ex in normal_examples]
    fraud_vals = [ex['data'][col] for ex in fraud_examples]
    
    normal_range = f"[{min(normal_vals):.2f}, {max(normal_vals):.2f}]"
    fraud_range = f"[{min(fraud_vals):.2f}, {max(fraud_vals):.2f}]"
    
    print(f"{col:<8} {normal_range:>20} {fraud_range:>20}")

# Statistical comparison
print("\n" + "="*80)
print("STATISTICAL COMPARISON (All 5 Examples)")
print("="*80)

fraud_data = df[df['Class']==1][v_cols + ['Amount']]
normal_data = df[df['Class']==0][v_cols + ['Amount']]

print(f"\n{'Feature':<8} {'Normal Mean':>12} {'Fraud Mean':>12} {'Example Normal':>15} {'Example Fraud':>15}")
print("-" * 70)

for col in v_cols[:10]:  # First 10 V features
    normal_mean = normal_data[col].mean()
    fraud_mean = fraud_data[col].mean()
    
    # Average of our examples
    example_normal = np.mean([ex['data'][col] for ex in normal_examples])
    example_fraud = np.mean([ex['data'][col] for ex in fraud_examples])
    
    print(f"{col:<8} {normal_mean:>12.4f} {fraud_mean:>12.4f} {example_normal:>15.4f} {example_fraud:>15.4f}")

print(f"\nAmount   {normal_data['Amount'].mean():>12.2f} {fraud_data['Amount'].mean():>12.2f}")

# Save examples to JSON for app.py integration
output_data = {
    'fraud_examples': fraud_examples,
    'normal_examples': normal_examples
}

with open('example_transactions.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\n" + "="*80)
print(f"✅ Saved {len(fraud_examples)} fraud + {len(normal_examples)} normal = {len(fraud_examples) + len(normal_examples)} total examples to example_transactions.json")
print("="*80)

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
