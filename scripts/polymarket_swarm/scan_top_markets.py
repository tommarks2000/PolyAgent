"""Scan top promising markets with Perplexity research."""
from datetime import datetime, timezone
from polymarket_client import PolymarketClient
from enhanced_research import EnhancedResearchClient

print("=" * 70)
print("TOP MARKET SCANNER - Perplexity Enhanced")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Fetch markets
with PolymarketClient() as pm:
    markets = pm.fetch_tradeable_markets(limit=200)

# Pre-filter for interesting markets
candidates = []
for m in markets:
    q = (m.question or "").lower()

    # Skip crypto/sports
    if any(x in q for x in ["bitcoin", "ethereum", "crypto", "solana", "xrp",
                             "nfl", "nba", "champion", "verstappen", "norris"]):
        continue

    # Calculate days until resolution
    days_left = 365
    if m.endDate:
        try:
            end = datetime.fromisoformat(m.endDate.replace('Z', '+00:00'))
            days_left = (end - datetime.now(timezone.utc)).days
        except:
            pass

    # Criteria for analysis:
    # 1. Extreme price (potential mispricing)
    # 2. Good volume
    # 3. Resolves within 60 days
    yes_price = m.yes_price or 0.5
    vol = m.volume or 0

    if days_left <= 60 and vol > 100000:
        # Calculate interest score
        price_extremity = abs(yes_price - 0.5) * 2
        interest = price_extremity * (vol / 1000000) * (60 / max(days_left, 1))

        candidates.append({
            'market': m,
            'days_left': days_left,
            'interest': interest,
        })

# Sort by interest
candidates.sort(key=lambda x: x['interest'], reverse=True)

# Take top 10
top_markets = [c['market'] for c in candidates[:10]]

print(f"\nAnalyzing top {len(top_markets)} high-interest markets:\n")
for i, m in enumerate(top_markets, 1):
    c = next(c for c in candidates if c['market'] == m)
    print(f"{i}. {m.question[:55]}...")
    print(f"   YES: {m.yes_price:.1%} | Days: {c['days_left']} | Vol: ${m.volume:,.0f}")

# Deep research
results = []
with EnhancedResearchClient() as client:
    print(f"\n{'='*70}")
    print("DEEP RESEARCH WITH PERPLEXITY")
    print(f"{'='*70}\n")

    for i, market in enumerate(top_markets, 1):
        print(f"[{i}/{len(top_markets)}] Researching: {market.question[:50]}...")

        result = client.research_market(market)

        # Calculate if there's edge
        edge = result.edge_estimate
        rec = "YES" if edge > 0 else "NO" if edge < 0 else "SKIP"

        results.append({
            'market': market,
            'research': result,
            'edge': edge,
            'recommendation': rec,
        })

        if abs(edge) >= 5:
            print(f"   >>> POTENTIAL EDGE: {edge:+.1f}% ({rec})")
        else:
            print(f"   Edge: {edge:+.1f}% (below threshold)")

# Summary
print(f"\n{'='*70}")
print("ANALYSIS SUMMARY")
print(f"{'='*70}\n")

# Sort by absolute edge
results.sort(key=lambda x: abs(x['edge']), reverse=True)

opportunities = [r for r in results if abs(r['edge']) >= 8]

if opportunities:
    print(f"Found {len(opportunities)} opportunities with 8%+ edge:\n")

    for r in opportunities:
        m = r['market']
        res = r['research']

        print(f"{'='*60}")
        print(f"MARKET: {m.question}")
        print(f"{'='*60}")
        print(f"Current: YES {m.yes_price:.1%} / NO {m.no_price:.1%}")
        print(f"Fair Value: {res.estimated_probability:.1%}" if res.estimated_probability else "")
        print(f"")
        print(f">>> RECOMMENDATION: {r['recommendation']}")
        print(f">>> EDGE: {r['edge']:+.1f}%")
        print(f">>> CONFIDENCE: {res.perplexity_confidence:.1%}")
        print(f"")
        print(f"Volume: ${m.volume:,.0f}")
        print(f"Liquidity: ${m.liquidity:,.0f}")
        print(f"")

        if res.key_factors:
            print("Key Factors:")
            for f in res.key_factors[:3]:
                print(f"  - {f[:70]}...")

        if res.perplexity_citations:
            print(f"\nSources ({len(res.perplexity_citations)} citations)")

        print()
else:
    print("No opportunities with 8%+ edge found.")
    print("\nTop results by edge:")
    for r in results[:5]:
        m = r['market']
        print(f"  {r['edge']:+.1f}%: {m.question[:50]}...")

print(f"\n{'='*70}")
print(f"Scan complete: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*70}")
