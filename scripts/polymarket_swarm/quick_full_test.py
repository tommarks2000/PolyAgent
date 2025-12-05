"""Quick test showing the full system working together.

Tests:
1. Broader market fetching (all categories)
2. Order book analysis
3. Enhanced research with Perplexity
4. Combined output
"""
from datetime import datetime
from enhanced_orchestrator import EnhancedSwarmOrchestrator

print("=" * 70)
print("QUICK FULL SYSTEM TEST")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print()

with EnhancedSwarmOrchestrator(
    use_rag=False,
    use_perplexity=True,
    use_orderbook=True,
) as orch:
    print(f"Perplexity: {'ENABLED' if orch.use_perplexity else 'DISABLED'}")
    print(f"Order Book: {'ENABLED' if orch.use_orderbook else 'DISABLED'}")
    print()

    # Fetch ALL markets (not just categories)
    print("Fetching ALL tradeable markets...")
    markets = orch.polymarket.fetch_all_categories(
        limit=20,
        min_volume=10000,
        min_liquidity=5000,
    )

    print(f"Found {len(markets)} markets")
    print()

    # Show what we found
    print("Sample markets (showing category diversity):")
    for i, m in enumerate(markets[:8], 1):
        q = m.question[:55] if m.question else "Unknown"
        print(f"  {i}. {q}...")
    print()

    # Run analysis on just 5 markets (quick test)
    print(f"{'='*70}")
    print("ANALYZING 5 MARKETS WITH FULL STACK")
    print("(Perplexity research + Order Book analysis)")
    print(f"{'='*70}")
    print()

    opportunities = orch.find_opportunities(
        max_markets=5,
        min_edge=5.0,  # Lower threshold for demo
        min_confidence=0.5,
        deep_research=True,
    )

    # Show results
    print()
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print()

    if opportunities:
        for i, opp in enumerate(opportunities, 1):
            print(f"OPPORTUNITY #{i}")
            print(f"  Market: {opp.market_question[:60]}...")
            print(f"  Recommendation: {opp.recommendation.value}")
            print(f"  Edge: {opp.edge_percent:.1f}%")
            print(f"  Confidence: {opp.confidence:.1%}")
            print(f"  Current: YES {opp.current_yes_price:.1%} / NO {opp.current_no_price:.1%}")
            print(f"  Fair Value: {opp.estimated_fair_yes_price:.1%}")

            # Order book data
            if str(opp.market_id) in orch.orderbook_cache:
                ob = orch.orderbook_cache[str(opp.market_id)]
                print(f"  Order Book Health: {'GOOD' if ob.get('healthy') else 'CAUTION'}")
                print(f"  Spread: {ob.get('average_spread_percent', 0):.2f}%")
                print(f"  Trading Rec: {ob.get('recommendation', 'N/A')}")

            print()

        print(f"Found {len(opportunities)} opportunities")
    else:
        print("No opportunities found with 5%+ edge")
        print("(This is normal - markets are usually efficient)")

    # Show order book stats
    print()
    print(f"{'='*70}")
    print("ORDER BOOK ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print()

    if orch.orderbook_cache:
        healthy = sum(1 for ob in orch.orderbook_cache.values() if ob.get("healthy"))
        print(f"Markets analyzed: {len(orch.orderbook_cache)}")
        print(f"Healthy order books: {healthy}")
        print(f"Unhealthy (wide spread/thin): {len(orch.orderbook_cache) - healthy}")

        # Show order book details
        for market_id, ob in list(orch.orderbook_cache.items())[:3]:
            research = orch.research_cache.get(market_id)
            question = research.market_question[:40] if research else "Unknown"
            print(f"\n  {question}...")
            print(f"    Spread: {ob.get('average_spread_percent', 0):.2f}%")
            print(f"    Liquidity Score: {ob.get('average_liquidity_score', 0):.2f}")
            print(f"    Recommendation: {ob.get('recommendation', 'N/A')}")

print()
print(f"{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
