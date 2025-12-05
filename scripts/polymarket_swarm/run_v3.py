"""Run the enhanced v3 Polymarket scanner.

This is the main entry point for the comprehensive analysis system.
Combines: Web search, LLM swarm, order book analysis, hybrid scoring.
"""
from datetime import datetime
from orchestrator_v3 import EnhancedOrchestratorV3

print("=" * 70)
print("POLYMARKET SWARM TRADER v3.0")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print()
print("This system combines:")
print("  - Perplexity/OpenAI web search for real-time news")
print("  - Multi-model LLM consensus (swarm predictions)")
print("  - Order book liquidity analysis")
print("  - Hybrid scoring (edge + consensus + liquidity)")
print("  - Historical performance tracking")
print()

with EnhancedOrchestratorV3(
    use_web_search=True,
    use_llm_swarm=True,
    use_orderbook=True,
    use_hybrid_scoring=True,
    track_performance=True,
) as orch:
    # Run comprehensive scan
    opportunities = orch.find_opportunities(
        max_markets=30,      # Analyze 30 markets
        min_edge=8.0,        # Require 8%+ edge
        min_confidence=0.6,  # Require 60%+ confidence
        deep_analysis=True,  # Full analysis with web search + LLM
    )

    # Generate and print report
    print()
    print(orch.generate_report(opportunities))

    # Save results
    if opportunities:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v3_scan_{timestamp}.json"
        with open(filename, "w") as f:
            f.write(orch.to_json(opportunities))
        print(f"\nResults saved to: {filename}")

        # Summary
        print()
        print("=" * 50)
        print("TOP OPPORTUNITIES SUMMARY")
        print("=" * 50)
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"\n{i}. {opp.market_question[:50]}...")
            print(f"   {opp.recommendation.value} @ {opp.edge_percent:.1f}% edge, {opp.confidence:.0%} conf")
    else:
        print("\nNo opportunities found meeting criteria.")
        print("Try lowering min_edge or running with more markets.")

print()
print("=" * 70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
