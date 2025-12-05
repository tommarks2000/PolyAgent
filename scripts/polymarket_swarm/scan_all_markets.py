"""Comprehensive market scanner - scans ALL markets with order book analysis.

This is the main scanner that:
1. Scans ALL tradeable markets (not just specific categories)
2. Only excludes crypto and sports
3. Performs Perplexity deep research
4. Analyzes order book liquidity and slippage
5. Generates comprehensive trading recommendations
"""
from datetime import datetime, timezone
from enhanced_orchestrator import EnhancedSwarmOrchestrator

print("=" * 70)
print("POLYMARKET COMPREHENSIVE SCANNER v2.1")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print()
print("Features enabled:")
print("  - ALL market categories (except crypto/sports)")
print("  - Perplexity deep research")
print("  - Order book liquidity analysis")
print("  - Slippage estimation")
print()

with EnhancedSwarmOrchestrator(
    use_rag=False,
    use_perplexity=True,
    use_orderbook=True,
) as orch:
    print(f"Perplexity API: {'ENABLED' if orch.use_perplexity else 'DISABLED'}")
    print(f"Order Book Analysis: {'ENABLED' if orch.use_orderbook else 'DISABLED'}")
    print()

    # Use fetch_all_categories to get ALL markets
    print("Fetching ALL tradeable markets (excluding crypto/sports only)...")
    markets = orch.polymarket.fetch_all_categories(
        limit=100,  # Scan more markets
        min_volume=5000,  # Lower threshold
        min_liquidity=2000,
    )

    print(f"Found {len(markets)} eligible markets")
    print()

    # Show category breakdown
    categories = {}
    for m in markets:
        q = (m.question or "").lower()
        if any(x in q for x in ["trump", "biden", "president", "election", "congress", "senate"]):
            cat = "Politics/Elections"
        elif any(x in q for x in ["fed", "rate", "inflation", "gdp", "economic"]):
            cat = "Economy/Fed"
        elif any(x in q for x in ["ai", "openai", "google", "apple", "tech"]):
            cat = "Technology"
        elif any(x in q for x in ["ukraine", "russia", "china", "iran", "war", "ceasefire"]):
            cat = "Geopolitics"
        elif any(x in q for x in ["weather", "climate", "temperature"]):
            cat = "Weather/Climate"
        elif any(x in q for x in ["award", "oscar", "grammy", "movie"]):
            cat = "Entertainment"
        else:
            cat = "Other"

        categories[cat] = categories.get(cat, 0) + 1

    print("Market Categories Found:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print()

    # Run analysis with lower thresholds
    print(f"{'='*70}")
    print("ANALYZING MARKETS WITH DEEP RESEARCH + ORDER BOOK")
    print(f"{'='*70}")
    print()

    opportunities = orch.find_opportunities(
        max_markets=50,  # Analyze top 50
        min_edge=8.0,    # 8% minimum edge
        min_confidence=0.6,
        deep_research=True,
    )

    # Generate report
    print()
    print(orch.generate_report(opportunities))

    # Save results
    if opportunities:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_markets_scan_{timestamp}.json"
        with open(filename, "w") as f:
            f.write(orch.to_json(opportunities))
        print(f"\nResults saved to: {filename}")

    # Summary statistics
    print()
    print(f"{'='*70}")
    print("SCAN SUMMARY")
    print(f"{'='*70}")
    print(f"Markets Scanned: {len(markets)}")
    print(f"Opportunities Found: {len(opportunities)}")

    if opportunities:
        avg_edge = sum(o.edge_percent for o in opportunities) / len(opportunities)
        avg_conf = sum(o.confidence for o in opportunities) / len(opportunities)
        print(f"Average Edge: {avg_edge:.1f}%")
        print(f"Average Confidence: {avg_conf:.1%}")

        # Order book stats
        healthy_count = sum(
            1 for o in opportunities
            if str(o.market_id) in orch.orderbook_cache
            and orch.orderbook_cache[str(o.market_id)].get("healthy", False)
        )
        print(f"Markets with Healthy Order Books: {healthy_count}/{len(opportunities)}")

        # Best opportunities
        print()
        print("TOP 3 OPPORTUNITIES:")
        for i, opp in enumerate(opportunities[:3], 1):
            ob = orch.orderbook_cache.get(str(opp.market_id), {})
            print(f"  {i}. {opp.market_question[:50]}...")
            print(f"     {opp.recommendation.value} @ {opp.edge_percent:.1f}% edge")
            print(f"     Order Book: {ob.get('recommendation', 'N/A')}")
    else:
        print("No opportunities met the criteria.")
        print("Try lowering min_edge or min_confidence thresholds.")

print()
print(f"{'='*70}")
print(f"Scan complete: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*70}")
