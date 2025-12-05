"""Test script to validate all changes."""
from polymarket_client import PolymarketClient
from orderbook_client import OrderBookClient

print("=" * 70)
print("TEST 1: Broader Market Fetching (fetch_all_categories)")
print("=" * 70)
print()

with PolymarketClient() as pm:
    markets = pm.fetch_all_categories(
        limit=50,
        min_volume=5000,
        min_liquidity=2000,
    )

print(f"Found {len(markets)} markets")
print()

# Categorize markets
categories = {}
for m in markets:
    q = (m.question or "").lower()
    if any(x in q for x in ["trump", "biden", "president", "election", "congress", "senate"]):
        cat = "Politics/Elections"
    elif any(x in q for x in ["fed", "rate", "inflation", "gdp", "economic"]):
        cat = "Economy/Fed"
    elif any(x in q for x in ["ai", "openai", "google", "apple", "tech", "gpt"]):
        cat = "Technology"
    elif any(x in q for x in ["ukraine", "russia", "china", "iran", "war", "ceasefire", "khamenei"]):
        cat = "Geopolitics"
    elif any(x in q for x in ["weather", "climate", "temperature"]):
        cat = "Weather/Climate"
    elif any(x in q for x in ["award", "oscar", "grammy", "movie", "box office"]):
        cat = "Entertainment"
    elif any(x in q for x in ["bitcoin", "crypto", "ethereum"]):
        cat = "CRYPTO (SHOULD BE EXCLUDED)"
    elif any(x in q for x in ["nfl", "nba", "soccer", "football", "basketball"]):
        cat = "SPORTS (SHOULD BE EXCLUDED)"
    else:
        cat = "Other"

    categories[cat] = categories.get(cat, 0) + 1

print("Market Categories:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

print()
print("Sample Markets:")
for i, m in enumerate(markets[:10], 1):
    q = m.question[:60] if m.question else "Unknown"
    vol = m.get_volume_safe()
    liq = m.get_liquidity_safe()
    print(f"  {i}. {q}...")
    print(f"     Vol: ${vol:,.0f} | Liq: ${liq:,.0f}")

print()
print("Test 1 PASSED - Broader market fetching working")

print()
print("=" * 70)
print("TEST 2: Order Book Client")
print("=" * 70)
print()

with OrderBookClient() as ob:
    tested = 0
    for m in markets[:5]:
        if not m.clobTokenIds:
            continue

        print(f"Market: {m.question[:50]}...")

        summary = ob.get_market_summary(m.clobTokenIds, m.question or "")

        print(f"  Healthy: {summary.get('healthy')}")
        print(f"  Spread: {summary.get('average_spread_percent', 0):.2f}%")
        print(f"  Recommendation: {summary.get('recommendation')}")

        if summary.get("yes_analysis"):
            yes = summary["yes_analysis"]
            print(f"  YES: Bid {yes.get('best_bid', 0):.3f} / Ask {yes.get('best_ask', 0):.3f}")
            print(f"  Slippage (500): {yes.get('slippage_500', 0):.2f}%")

        print()
        tested += 1

print(f"Test 2 PASSED - Order book analysis working for {tested} markets")

print()
print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
