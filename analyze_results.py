"""Analyze the compression results to understand what's failing."""
import json
from pathlib import Path

# Load results
results = json.load(open('long_message_results/results.json'))

print("="*80)
print("COMPRESSION FAILURE ANALYSIS")
print("="*80)

print(f"\nOverall: {results['final_similarity']:.1%} similarity (target: 80%)")
print(f"Final compression: {results['final_compression']:.1%}")

print("\n" + "="*80)
print("ITERATION BREAKDOWN")
print("="*80)

for iter_data in results['iterations']:
    print(f"\n{'─'*80}")
    print(f"ITERATION {iter_data['iteration']}")
    print(f"{'─'*80}")
    print(f"Entropy Target: {iter_data['entropy_target']:.0%}")
    print(f"Nodes: {iter_data['nodes_kept']}/{iter_data['total_nodes']} ({iter_data['nodes_kept']/iter_data['total_nodes']:.1%})")
    print(f"Similarity: {iter_data['similarity_score']:.1%}")
    print(f"Compression: {iter_data['compression_ratio']:.1%}")
    print(f"\nImportance Retention: {iter_data['importance_retention']:.1%}")
    print(f"Entropy Retention: {iter_data['entropy_retention']:.1%}")
    print(f"Avg Node Importance: {iter_data['avg_node_importance']:.3f}")
    
    print(f"\nNodes by Type:")
    for node_type, count in iter_data['nodes_by_type'].items():
        print(f"  {node_type}: {count}")
    
    print(f"\nMissing Concepts ({len(iter_data['missing_concepts'])}):")
    for concept in iter_data['missing_concepts'][:5]:
        print(f"  - {concept}")
    if len(iter_data['missing_concepts']) > 5:
        print(f"  ... and {len(iter_data['missing_concepts']) - 5} more")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Analyze trends
similarities = [i['similarity_score'] for i in results['iterations']]
print(f"\nSimilarity Trend: {similarities[0]:.1%} → {similarities[-1]:.1%}")
if similarities[-1] < similarities[0]:
    print("⚠️  PROBLEM: Similarity DECREASED over iterations!")
elif similarities[-1] - similarities[0] < 0.1:
    print("⚠️  PROBLEM: Similarity barely improved (<10% gain)")

# Check if we're keeping enough nodes
final_iter = results['iterations'][-1]
if final_iter['nodes_kept'] / final_iter['total_nodes'] > 0.7:
    print(f"\n⚠️  PROBLEM: Keeping {final_iter['nodes_kept']}/{final_iter['total_nodes']} nodes (76%) but only {final_iter['similarity_score']:.1%} similarity")
    print("   This suggests the GRAPH EXTRACTION is losing critical information!")

# Check importance boosting effectiveness
importance_retentions = [i['importance_retention'] for i in results['iterations']]
print(f"\nImportance Retention Trend: {importance_retentions[0]:.1%} → {importance_retentions[-1]:.1%}")
if importance_retentions[-1] - importance_retentions[0] < 0.1:
    print("⚠️  PROBLEM: Importance boosting isn't working effectively")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n1. START WITH HIGHER ENTROPY for long messages:")
print("   - 1763 tokens → start at 50-60% entropy, not 40%")

print("\n2. INVESTIGATE GRAPH EXTRACTION:")
print("   - Check if critical information is being extracted as nodes")
print("   - Look at the actual nodes in iteration 1 - are they meaningful?")

print("\n3. FIX IMPORTANCE BOOSTING:")
print("   - Current boosting isn't helping - similarity barely improves")
print("   - Need better matching between missing concepts and nodes")

print("\n4. CHECK DECODER:")
print("   - Is it reconstructing accurately from the graph?")
print("   - Compare decoded message to original - what's different?")
