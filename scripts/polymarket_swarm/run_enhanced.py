"""Run enhanced Polymarket scan with Perplexity deep research."""
from datetime import datetime
from enhanced_orchestrator import EnhancedSwarmOrchestrator

print("=" * 70)
print("POLYMARKET SWARM TRADER v2.1 - PERPLEXITY ENHANCED")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

with EnhancedSwarmOrchestrator(use_rag=False, use_perplexity=True) as orch:
    # Run scan with lower thresholds to find more opportunities
    opportunities = orch.find_opportunities(
        max_markets=30,
        min_edge=8.0,       # Lower threshold
        min_confidence=0.6, # Lower confidence
        deep_research=True,
    )

    # Generate report
    print("\n")
    print(orch.generate_report(opportunities))

    # Save results
    if opportunities:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_opportunities_{timestamp}.json"
        with open(filename, "w") as f:
            f.write(orch.to_json(opportunities))
        print(f"\nResults saved to: {filename}")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
