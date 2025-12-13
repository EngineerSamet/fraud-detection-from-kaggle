"""
Test all 10 examples (5 fraud + 5 normal) and show predictions
"""
import json
import pandas as pd
from predict_fraud import FraudDetectionPipeline

# Load detector
print("Loading fraud detection model...")
detector = FraudDetectionPipeline()
detector.load_models()

# Load examples
with open('example_transactions.json', 'r') as f:
    examples = json.load(f)

print("\n" + "="*80)
print("TESTING ALL EXAMPLES")
print("="*80)

results = []

# Test fraud examples
print("\n" + "="*80)
print("FRAUD EXAMPLES (True Class: FRAUD)")
print("="*80)

for i, example in enumerate(examples['fraud_examples'], 1):
    transaction = example['data']
    result = detector.predict(transaction, threshold_type='f2_optimized')
    
    prob = result['fraud_probability']
    decision = (result['decision'] == 'FRAUD')
    anomaly = result['anomaly_score']
    
    print(f"\nFraud #{i}:")
    print(f"  Amount: ${transaction['Amount']:.2f}")
    print(f"  V1={transaction['V1']:.2f}, V3={transaction['V3']:.2f}, V7={transaction['V7']:.2f}")
    print(f"  → Probability: {prob:.2%}")
    print(f"  → Anomaly Score: {anomaly:.4f}")
    print(f"  → Decision: {'🚨 FRAUD' if decision else '✅ SAFE'}")
    print(f"  → Status: {'✅ CORRECT' if decision else '❌ MISSED (False Negative)'}")
    
    results.append({
        'Type': 'Fraud',
        'Example': f'Fraud #{i}',
        'Amount': transaction['Amount'],
        'V1': transaction['V1'],
        'Probability': prob,
        'Anomaly': anomaly,
        'Decision': decision,
        'Correct': decision
    })

# Test normal examples
print("\n" + "="*80)
print("NORMAL EXAMPLES (True Class: NORMAL)")
print("="*80)

for i, example in enumerate(examples['normal_examples'], 1):
    transaction = example['data']
    result = detector.predict(transaction, threshold_type='f2_optimized')
    
    prob = result['fraud_probability']
    decision = (result['decision'] == 'FRAUD')
    anomaly = result['anomaly_score']
    
    print(f"\nNormal #{i}:")
    print(f"  Amount: ${transaction['Amount']:.2f}")
    print(f"  V1={transaction['V1']:.2f}, V3={transaction['V3']:.2f}, V7={transaction['V7']:.2f}")
    print(f"  → Probability: {prob:.2%}")
    print(f"  → Anomaly Score: {anomaly:.4f}")
    print(f"  → Decision: {'🚨 FRAUD' if decision else '✅ SAFE'}")
    print(f"  → Status: {'✅ CORRECT' if not decision else '❌ FALSE ALARM (False Positive)'}")
    
    results.append({
        'Type': 'Normal',
        'Example': f'Normal #{i}',
        'Amount': transaction['Amount'],
        'V1': transaction['V1'],
        'Probability': prob,
        'Anomaly': anomaly,
        'Decision': decision,
        'Correct': not decision
    })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

fraud_results = results_df[results_df['Type'] == 'Fraud']
normal_results = results_df[results_df['Type'] == 'Normal']

fraud_correct = fraud_results['Correct'].sum()
normal_correct = normal_results['Correct'].sum()

print(f"\nFraud Detection (Recall):")
print(f"  Caught: {fraud_correct}/5 ({fraud_correct/5*100:.1f}%)")
print(f"  Missed: {5-fraud_correct}/5 ({(5-fraud_correct)/5*100:.1f}%)")

print(f"\nNormal Approval (Precision):")
print(f"  Approved: {normal_correct}/5 ({normal_correct/5*100:.1f}%)")
print(f"  False Alarms: {5-normal_correct}/5 ({(5-normal_correct)/5*100:.1f}%)")

print(f"\nOverall Accuracy: {(fraud_correct + normal_correct)}/10 ({(fraud_correct + normal_correct)/10*100:.1f}%)")

# Show probability distribution
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION")
print("="*80)

print(f"\n{'Example':<15} {'True Class':<10} {'Probability':>12} {'Decision':>12} {'Status':>10}")
print("-" * 65)

for _, row in results_df.iterrows():
    status = '✅' if row['Correct'] else '❌'
    decision = 'FRAUD' if row['Decision'] else 'SAFE'
    print(f"{row['Example']:<15} {row['Type']:<10} {row['Probability']:>11.2%} {decision:>12} {status:>10}")

print("\n" + "="*80)
print("💡 INSIGHTS")
print("="*80)

# Find subtle frauds (low probability but still fraud)
subtle_frauds = fraud_results[fraud_results['Probability'] < 0.80]
if len(subtle_frauds) > 0:
    print(f"\n⚠️ SUBTLE FRAUDS ({len(subtle_frauds)} found):")
    for _, row in subtle_frauds.iterrows():
        print(f"  {row['Example']}: {row['Probability']:.2%} - V1={row['V1']:.2f}")
        print(f"    → {'Caught' if row['Decision'] else 'MISSED'} by F2-optimized threshold (0.60)")

# Find strong frauds (high probability)
strong_frauds = fraud_results[fraud_results['Probability'] >= 0.90]
if len(strong_frauds) > 0:
    print(f"\n🔥 OBVIOUS FRAUDS ({len(strong_frauds)} found):")
    for _, row in strong_frauds.iterrows():
        print(f"  {row['Example']}: {row['Probability']:.2%} - V1={row['V1']:.2f}")
        print(f"    → Extreme V values make this unmistakable")

# Check for false alarms
false_alarms = normal_results[normal_results['Decision'] == True]
if len(false_alarms) > 0:
    print(f"\n❌ FALSE ALARMS ({len(false_alarms)} found):")
    for _, row in false_alarms.iterrows():
        print(f"  {row['Example']}: {row['Probability']:.2%} - V1={row['V1']:.2f}")
        print(f"    → Normal transaction incorrectly flagged as fraud")
else:
    print("\n✅ NO FALSE ALARMS - Perfect precision on these examples!")
