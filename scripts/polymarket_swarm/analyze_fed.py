from polymarket_client import PolymarketClient
client = PolymarketClient()
markets = client.fetch_tradeable_markets(limit=300)

print('=' * 90)
print('FED RATE DECISION MARKETS - Expires Dec 9, 2025 (4 days)')
print('=' * 90)
print()

fed_markets = []
for m in markets:
    if 'fed rate' in m.question.lower() and '2025' in m.question:
        fed_markets.append(m)

for m in sorted(fed_markets, key=lambda x: x.yes_price or 0, reverse=True):
    yes = m.yes_price or 0.5
    print(f'{m.question}')
    print(f'  YES: {yes*100:5.1f}% | NO: {(1-yes)*100:5.1f}%')
    print(f'  Volume: ${m.volume:,.0f} | Liquidity: ${m.liquidity:,.0f}')
    print()
