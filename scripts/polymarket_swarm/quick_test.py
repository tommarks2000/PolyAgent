"""Quick test of enhanced research on select markets."""
from datetime import datetime
from polymarket_client import PolymarketClient
from enhanced_research import EnhancedResearchClient

print("=" * 70)
print("QUICK TEST: Enhanced Perplexity Research")
print("=" * 70)

# Get a few interesting markets
with PolymarketClient() as pm:
    markets = pm.fetch_tradeable_markets(limit=100)

# Filter for interesting ones
test_markets = []
for m in markets:
    q = (m.question or "").lower()
    # Skip crypto/sports
    if any(x in q for x in ["bitcoin", "ethereum", "crypto", "nfl", "nba", "champion"]):
        continue
    # Look for geopolitical/political
    if any(x in q for x in ["ceasefire", "putin", "trump", "fed", "khamenei", "maduro"]):
        test_markets.append(m)
    if len(test_markets) >= 3:
        break

if not test_markets:
    test_markets = markets[:3]

print(f"\nTesting {len(test_markets)} markets with Perplexity deep research:")
for m in test_markets:
    print(f"  - {m.question[:60]}... (YES: {m.yes_price:.1%})")

# Test enhanced research
with EnhancedResearchClient() as client:
    print(f"\nPerplexity API: {'ENABLED' if client.has_perplexity else 'DISABLED'}")

    for i, market in enumerate(test_markets, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(test_markets)}] {market.question}")
        print(f"Current Price: YES {market.yes_price:.1%} / NO {market.no_price:.1%}")
        print(f"{'='*60}")

        result = client.research_market(market)

        print(f"\nResearch Results:")
        print(f"  Sources: {result.sources_used}")
        print(f"  News Articles: {len(result.news_articles)}")
        print(f"  News Sentiment: {result.news_sentiment:.2f}")
        print(f"  Combined Sentiment: {result.combined_sentiment:.2f}")
        print(f"  Perplexity Confidence: {result.perplexity_confidence:.1%}")

        if result.estimated_probability:
            print(f"\n  >>> PROBABILITY ESTIMATE: {result.estimated_probability:.1%}")
            print(f"  >>> EDGE: {result.edge_estimate:+.1f}%")

        if result.key_factors:
            print(f"\n  Key Factors:")
            for f in result.key_factors[:3]:
                print(f"    - {f[:70]}...")

        if result.perplexity_citations:
            print(f"\n  Citations ({len(result.perplexity_citations)}):")
            for c in result.perplexity_citations[:3]:
                print(f"    - {c[:60]}...")

        if result.perplexity_analysis:
            print(f"\n  Analysis Preview:")
            print(f"    {result.perplexity_analysis[:300]}...")

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
